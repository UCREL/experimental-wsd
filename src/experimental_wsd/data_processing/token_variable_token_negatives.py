"""
All of these Data Processing utility functions are for data samples that have 
a format like so, whereby the task is given a text with a given token offset 
(can encompass more than one token as it is an offset) find the most similar label 
from a list of labels whereby ahead of time you know the positive labels and 
negative labels. In essence a semantic similarity task at the token level.

**The number of negative labels is variable, e.g. one sample can contain 1 
negative, whereas another could contain 2/3/4/5**

Each sample contains something like:

`text`: A list of token texts that represents the contextualized text. list[str].
    Example ["Did", "you", "have", "a", "good", "or", "bad", "day"]
`token_offset`: A tuple of token start and end indexes tuple[int, int]. 
    Example (6,8).
`positive_labels`: A list of positive labels for the token offset.
    list[str]. Example ["bad day%2"]
`negative_labels`: A list of negative labels for the token offset. Can be 
    an empty list if no negative labels exist (these negative labels tend to 
    be "hard" negatives as they are very close to the positive label but not quiet 
    the same). Example ["good_day%4", "bad_month%3"]
"""