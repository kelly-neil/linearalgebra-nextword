## Instructions

Install numpy using `python3 -m pip install numpy`or `python -m pip install numpy`. (if the v2 file is being used)

Then run any of the following scripts:
- `next_word_suggester_v2.py` (not final and still experimental), which uses embeddings for smarter suggestions.
- `next_word_suggester_v1.py` initial version which uses probability matrices.
- `suggester_gui.py` runs an interactive app of the suggester, rather than in the terminal.

You can run any of the three with `python3 <filename>.py` or `python <filename>.py`.

You will also need one or more corpus files. Simply upload them in the project directory with the name `corpus_<name>.py`

## Credits

The corpus `corpus_dailydialogs.txt` is extracted from [DailyDialog](https://aclanthology.org/I17-1099/) which is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
