"""
Live GUI (multi-line) for the next-word suggester. tkinter is in the standard library.
Put this file in the same folder as next_word_suggester.py, then run it.
"""
import re
import tkinter as tk
from next_word_suggester_v1 import suggest


class App:
    def __init__(self, root):
        root.title("Next-Word Suggester")

        self.text = tk.Text(root, font=("Arial", 14), width=50, height=8,
                            wrap="word", undo=True)
        self.text.pack(padx=10, pady=10)
        self.text.focus()

        # <<Modified>> fires on any change (typing, backspace, paste, button click).
        # KeyRelease / ButtonRelease cover moving the cursor without changing text.
        self.text.bind("<<Modified>>", self.on_modified)
        self.text.bind("<KeyRelease>", lambda e: self.refresh())
        self.text.bind("<ButtonRelease-1>", lambda e: self.refresh())

        bar = tk.Frame(root)
        bar.pack(padx=10, pady=(0, 10), fill="x")
        self.buttons = []
        for _ in range(3):
            b = tk.Button(bar, font=("Arial", 14), width=12)
            b.pack(side="left", expand=True, fill="x", padx=2)
            self.buttons.append(b)
        self.refresh()

    def on_modified(self, event=None):
        if self.text.edit_modified():
            self.refresh()
        self.text.edit_modified(False)   # reset so the next change fires again

    def before_cursor(self):
        """Text from the start up to the cursor, with newlines treated as spaces."""
        return self.text.get("1.0", "insert").replace("\n", " ")

    def refresh(self):
        try:
            results = suggest(self.before_cursor())
        except Exception as e:
            print(f"Suggestion error: {e}")
            results = []  # Fallback to empty list so the window doesn't crash
            
        for b, word in zip(self.buttons, results + [""] * 3):
            b.config(text=word, command=lambda w=word: self.pick(w),
                     state="normal" if word else "disabled")

    def pick(self, word):
        """Replace the partial word before the cursor with the chosen suggestion."""
        partial = re.split(r"[ \n]", self.text.get("1.0", "insert"))[-1]
        if partial:
            self.text.delete(f"insert-{len(partial)}c", "insert")
        self.text.insert("insert", word + " ")
        self.text.focus()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
