import pypdfium2 as pdfium
import tiktoken
import openai
import sys
from time import sleep
import os

model = 'gpt-3.5-turbo-16k'
input_token_limit = 10_000 # Leave 6k for prompt and response
total_token_limit = 16_000
cost_per_1k_input = 0.003
cost_per_1k_output = 0.004

prompt_de = ('Du bist UniGPT, ein Sprachmodell, dessen Aufgabe das Zusammenfassen von Vorlesungsinhalten als Lernhilfe für Studenten zur Klausurvorbereitung ist. ' +
    'Fasse die wichtigsten Inhalte der folgenden Vorlesungsfolien als Lernkarten zusammen, verwende dazu generell Stichpunkte, erläutere Begriffsdefinitionen und andere hochrelevante Punkte aber ausführlicher: ')
prompt = prompt_de
prompt_length = 100

def extract_text(pdf_path: str) -> tuple[str, ...]:
    pages = []
    pdf = pdfium.PdfDocument(pdf_path)
    for page in pdf:
        textpage = page.get_textpage()
        pages.append(textpage.get_text_range())

    return tuple(pages)

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    encoded_text = encoding.encode(text)
    return len(encoded_text)

def split_pages(pages: tuple[str, ...], num_tokens: int) -> tuple[tuple[str, ...], ...]:
    num_batches = num_tokens / input_token_limit + 1
    tokens_per_batch = num_tokens / num_batches

    batches = []
    curr_batch = []
    curr_batch_size = 0
    for page in pages:
        next_page_size = count_tokens(page)

        if curr_batch_size + next_page_size > input_token_limit or curr_batch_size >= tokens_per_batch:
            # TODO: curr_batch empty -> next page breaks token limit by itself
            batches.append(curr_batch)
            curr_batch = []
            curr_batch_size = 0
        
        if curr_batch_size + next_page_size <= input_token_limit:
            curr_batch.append(page)
            curr_batch_size += next_page_size

    if curr_batch:
        batches.append(curr_batch)
    
    return tuple(batches)

def call_api(text: str) -> str:

    req_content = prompt + text
    completion = openai.ChatCompletion.create(model=model, messages=[{'role': 'user', 'content': req_content}], request_timeout=1200, max_tokens=(total_token_limit - input_token_limit - prompt_length))

    resp = completion.choices[0].message.content
    print(resp)

    return resp

def confirm_prompt(question: str) -> bool:
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ").casefold()
    return (reply == "y")

def cost_disclaimer(num_batches: int):
    input_cost_per_batch = (input_token_limit/1000) * cost_per_1k_input
    output_cost_per_batch = ((total_token_limit - input_token_limit) / 1000) * cost_per_1k_output
    total_cost_per_batch = input_cost_per_batch + output_cost_per_batch

    print(f"About to call the openAI API. This is expected to cost up to ${num_batches * total_cost_per_batch}.")
    if not confirm_prompt("Continue?"):
        print("Exiting")
        exit

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print('Could not find openAI API key. Please provide it as an environment variable named "OPENAI_API_KEY".')

pages = []
for pdf_path in sys.argv[1:]:
    pages += extract_text(pdf_path)

num_tokens = count_tokens('\n'.join(pages))
if num_tokens > input_token_limit:
    batches = split_pages(pages, num_tokens)
    print(f'The specified file(s) is/are too long for a single request and has/have been split into {len(batches)} batches.')
else:
    batches = (pages,)

cost_disclaimer(len(batches))

print("Calling openAI API, this may take several minutes.")

summary = ''
for i, batch in enumerate(batches):
    print(f'Batch {i+1}/{len(batches)}')
    summary += call_api('\n'.join(batch)) + '\n'
    sleep(60) # Rate limits

with open('summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("Summarization finished. Wrote summary to 'summary.txt'")