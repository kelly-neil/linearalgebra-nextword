"""
Live GUI (multi-line) for the next-word suggester. tkinter is in the standard library.
This GUI uses next_word_suggester_v1.py from the same project folder.
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
        self.text.bind("<KeyRelease>", self.refresh)
        self.text.bind("<ButtonRelease-1>", self.refresh)

        bar = tk.Frame(root)
        bar.pack(padx=10, pady=(0, 10), fill="x")
        self.buttons = []
        for _ in range(3):
            b = tk.Button(bar, font=("Arial", 14), width=12)
            b.pack(side="left", expand=True, fill="x", padx=2)
            self.buttons.append(b)
        self.status = tk.Label(root, anchor="w", foreground="red")
        self.status.pack(padx=10, pady=(0, 10), fill="x")
        self.refresh()

    def on_modified(self, event=None):
        if event is not None and event.widget is not self.text:
            return
        if self.text.edit_modified():
            self.refresh()
        self.text.edit_modified(False)   # reset so the next change fires again

    def full_text_at_cursor(self):
        """Text from start through the end of the word the cursor is inside."""
        before = self.text.get("1.0", "insert")
        after = self.text.get("insert", "end")
        word_tail = re.split(r"[ \n]", after)[0]
        return (before + word_tail).replace("\n", " ")

    def refresh(self, event=None):
        if event is not None and event.widget is not self.text:
            return
        try:
            results = suggest(self.full_text_at_cursor())
        except Exception as e:
            message = f"Suggestion error: {e}"
            print(message)
            self.status.config(text=message)
            results = []
        else:
            self.status.config(text="")

        for b, word in zip(self.buttons, results + [""] * 3):
            b.config(text=word, command=lambda w=word: self.pick(w),
                     state="normal" if word else "disabled")

    def pick(self, word):
        """Replace the whole word around the cursor with the chosen suggestion."""
        before = re.split(r"[ \n]", self.text.get("1.0", "insert"))[-1]
        after = re.split(r"[ \n]", self.text.get("insert", "end"))[0]
        start = f"insert-{len(before)}c"
        end = f"insert+{len(after)}c"
        self.text.delete(start, end)
        self.text.insert("insert", word + " ")
        self.text.focus()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
