"""
All of these Data Processing utility functions are for data samples that have 
a format like so, whereby the task is given a text with a given token offset 
(can encompass more than one token as it is an offset) find the most similar label 
from a list of labels whereby ahead of time you know the positive label and 
negative labels. In essence a semantic similarity task at the token level.

**In all cases the number of positive labels per sample is expected to be 1**

**The number of negative labels is static, e.g. all samples must contain 1/2/3 
or None negative labels**

Each sample contains something like:

`text`: A list of token texts that represents the contextualized text. list[str].
    Example ["Did", "you", "have", "a", "good", "or", "bad", "day"]
`token_offset`: A tuple of token start and end indexes tuple[int, int]. 
    Example (6,8).
`positive_labels`: The positive label for the token offset. str.
    Example "bad day%2"
`negative_labels`: A list of negative labels for the token offset. Can be 
    an empty list if no negative labels exist (these negative labels tend to 
    be "hard" negatives as they are very close to the positive label but not quiet 
    the same). Example ["good_day%4", "bad_month%3"]
"""