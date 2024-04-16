import openai
import tiktoken
from tqdm.auto import trange, tqdm
import json
from tqdm import tqdm
from types import NoneType
import sys
import ast
import copy
import argparse

errors = {}
error_lineno = None
lines = None
trace_lines = []
last_state = None

TOTAL_CORRECT_ANSWERS = 0
FAILED_RUNS = 0


class ChainOfCode():
    def __init__(self, args, answer):
        self.engine = args.model_name
        self.answer_token = 'Answer: '
        self.correct_answer = answer
        self.code_start_token = "# CODE START"
        self.code_end_token = "# CODE END"
        self.max_tokens_generation = args.max_tokens_generation
        self.max_tokens_lmulator = args.max_tokens_lmulator
        self.num_trials = args.num_trials
        self.encoder = tiktoken.encoding_for_model(self.engine)
        
        self.progress_bar = None
                
        with open('models/prompts/CoC-trace-prompt.txt', 'r') as f:
            self.coc_trace_prompt = f.read().strip()
        
        with open('models/prompts/CoC-code-generation-prompt.txt', 'r') as f:
            self.coc_generation_prompt = f.read().strip()
        
        
    def query_llm(self, prompt, max_tokens, stop=None, temperature=0):
        """
        Give the prompt to the LLM and get response
        """
        assert type(prompt)
        
        if 'instruct' in self.engine:
            response = openai.Completion.create(prompt=prompt, model=self.engine, max_tokens=max_tokens, temperature=temperature, stop=stop)
            response_text = response.choices[0]["text"].strip()
            return response_text
        else:        
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(messages=messages, model=self.engine, max_tokens=max_tokens, temperature=temperature, stop=stop)
            return response['choices'][0]['message']['content'].strip()
    
    
    def print_result(self, method, response, answer):
        global TOTAL_CORRECT_ANSWERS
        print("#### Method ####")
        print(method)
        print("#### Full Response ####")
        print(response)
        print("#### Model Answer ####")
        print(answer)
        print("#### Correct Answer ####")
        print(self.correct_answer)
        if str(answer).strip().lower() == str(self.correct_answer).strip().lower():
            TOTAL_CORRECT_ANSWERS += 1
        
            
    def get_delta_state(self, state, last_state):
        """
        This method indicates the difference between the current state and the
        last state. So it focses on the differences between two consecutive states.

        Returns a dictionary that contains info about the changes between two states.
        """
        delta_state = {}
        for key, val in state.items():
            if key not in last_state or val != last_state[key]:
                delta_state[key] = val
        return delta_state


    def get_state(self, frame):
        """
        This method 'captures' and returns the local variables that currently in
        the frame that we pass as an argument
        """
        state = {}
        for key, item in frame.f_locals.items():
            if isinstance(item, (bool, str, int, float, tuple, list, set, dict, NoneType)):
                state[key] = item
        return state


    def show_trace(self, frame, event, arg):
        # Declare these global variable first
        global errors
        global error_lineno
        global lines
        global trace_lines
        global last_state

        # The LLM-generated code will be wrapped around in the get_answer function call.
        # If we don't filter by "get_answer", we got a bunch of random exception from colab
        if frame.f_code.co_name != "get_answer":
            return
        
        lineno = frame.f_lineno - 1
        # Running a certain line
        # If the program is about to execute a new line of code
        if event == "line":
            current_line = lines[lineno]
            if current_line.strip() in ["try:", "except:", "pass"]:
                pass
            elif current_line.strip() == "return answer":
                assert lineno == len(lines) - 2, "return answer is at the wrong line" # Second to last line
                state = self.get_state(frame)
                assert last_state is not None
                delta_state = self.get_delta_state(state, last_state)
                trace_lines.append(f"delta state: {delta_state}")
                # Append the final state
                trace_lines.append(f"final state: {state}")
            elif lineno not in errors:
                # We previous indent 2 spaces
                assert current_line[:2] == "  ", f"Python: actual line to run doesn't have two leading spaces: {current_line} {lines}"
                # Now we revert back
                current_line = current_line[2:]

                state = self.get_state(frame)
                delta_state = None
                if last_state is None:
                    delta_state = None
                else:
                    delta_state = self.get_delta_state(state, last_state)
                last_state = copy.deepcopy(state)

                if delta_state is None:
                    trace_lines.append("state: {}")
                else:
                    trace_lines.append(f"delta state: {delta_state}")
                trace_lines.append(f"line: {current_line}")
            else:
                # We previous indent 4 spaces
                assert current_line[:4] == "    ", f"LLM: actual line to run doesn't have four leading spaces: {current_line} {lines}"
                # Now we revert back
                current_line = current_line[4:]
                # When LLM excutes, remove any trailing space at the beginning

                state = self.get_state(frame)
                delta_state = None
                if last_state is None:
                    delta_state = None
                else:
                    delta_state = self.get_delta_state(state, last_state)
                last_state = copy.deepcopy(state)

                if delta_state is None:
                    trace_lines.append("state: {}")
                else:
                    trace_lines.append(f"delta state: {delta_state}")
                trace_lines.append(f"line: {current_line}")

                # Due to the context length constraint, only feed in the last three lines of the trace.
                prompt = self.coc_trace_prompt + "\n" + "\n".join(trace_lines[-3:]) + "\n" + "delta state:"

                token_length = len(self.encoder.encode(prompt))

                llm_result = self.query_llm(prompt, max_tokens=self.max_tokens_lmulator, stop=["\nline:"])
                program_state_str = llm_result.strip()
                try:
                    new_program_state = ast.literal_eval(program_state_str)
                    assert isinstance(new_program_state, dict), "new program state is not a valid dict"
                    # Actually update the local variables with the new program state
                    frame.f_locals.update(new_program_state)
                except Exception as e:
                    raise e

        elif event == "exception":
            # Only capture the lowest level exception AND if this exception hasn't been "fixed" before, i.e. this line hasn't be sandwiched by try/except yet.
            if error_lineno is None and lineno not in errors:
                error_lineno = lineno

        return self.show_trace

    
    def evaluate_coc(self, query):
        """
        Evaluates the response from the CoC approach.

        Queries LLM
        Parses Response to extract Generated Code (between 'CODE_START_TOKEN' and 'CODE_END_TOKEN')
        Creates the 'code_to_run' obj that contains the code
        'max_trials': times that the loop can run
        """
        global errors
        global error_lineno
        global lines
        global trace_lines
        global last_state
        global FAILED_RUNS
        
        
        # print(self.coc_generation_prompt + "\n\n" + query)
        coc_response = self.query_llm(self.coc_generation_prompt + "\n\n" + query, max_tokens=self.max_tokens_generation)
        # print(coc_response)
        if self.code_start_token in coc_response and self.code_end_token in coc_response:
            code_to_run = coc_response.split(self.code_start_token)[1].split(self.code_end_token)[0].strip()
        else:
            FAILED_RUNS += 1
            return
            
        answer = None
        # Wrap the code inside the get_answer function call
        code_to_run_temp = code_to_run.split("\n")
        # code_to_run_temp = self.fix_indentation(code_to_run).split("\n")
        code_to_run = "\n".join(["  " + l for l in code_to_run_temp])
        code_to_run = f"""def get_answer():
{code_to_run}
  return answer
answer = get_answer()"""
        
        # print(code_to_run)
        
        lines = code_to_run.split("\n")
        local_vars = locals()

        for num_trial in range(self.num_trials):
            if sys.gettrace() is None: sys.settrace(self.show_trace)
            assert sys.gettrace() is not None, "get trace is None"
            try:
                # answer will be populated by exec function.
                exec(code_to_run, globals(), local_vars)
                coc_answer = local_vars["answer"]
                assert coc_answer is not None
                break
            except Exception as e:
                if error_lineno is None:
                    FAILED_RUNS += 1
                    return
                
                # Update errors
                line = lines[error_lineno]
                errors[error_lineno + 1] = line

                # Update lines and code_to_run
                num_indent = len(line) - len(line.lstrip())
                lines[error_lineno] = " " * 2 + lines[error_lineno]
                lines.insert(error_lineno, " " * num_indent + "try:")
                lines.insert(error_lineno + 2, " " * num_indent + "except:")
                lines.insert(error_lineno + 3, " " * (num_indent + 2) + "pass")
                code_to_run = "\n".join(lines)

                # Reset error_lineno and trace_lines
                error_lineno = None
                trace_lines = []
                last_state = None

        self.print_result('CoC', coc_response, coc_answer)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--num_trials', type=int, default=3)
    parser.add_argument('--max_tokens_generation', type=int, default=1024)
    parser.add_argument('--max_tokens_lmulator', type=int, default=32)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    file_path = f"data/{args.dataset_name}/{args.dataset_name}_for_CoC.json"
    with open(file_path, 'r') as f:
      data = json.load(f)  # Load data from the file
    
    for item in data:
        
        CoC = ChainOfCode(args, item['answer'])
        query = f"""
    Q: {item['question']}
    """.strip()
        sys.settrace(CoC.show_trace)
        CoC.evaluate_coc(query)
        
    print(f"Total correct answers: {TOTAL_CORRECT_ANSWERS}")
    print(f"Total failed runs: {FAILED_RUNS}")
