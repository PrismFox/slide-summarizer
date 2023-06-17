# How to use
### Install requirements
> pip install -r requirements.txt -U

### Set your OpenAI key as an environment variable
> export OPENAI_API_KEY = "sk-..."

### Run the script
> python summarize.py ./slides/*.pdf
or
> python summarize.py ./slides/chapter1.pdf ./slides/chapter2.pdf ./slides/chapter3.pdf

# Known limitations
- The script only extracts text from the slides. The heavier the slides depend on images the worse the results will get.
- Due to how slides are split into batches, output formatting may drastically change from one section to the next.
- Context is limited to the same batch.

A lot of this will improve when using GPT-4 due to higher token limits and the possibility of multimodal input.
