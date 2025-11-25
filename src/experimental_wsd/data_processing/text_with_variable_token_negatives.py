"""
All of these Data Processing utility functions are for data samples that have
a format like so, whereby the task is given a text with a list of token offsets
for each token offsets (can encompass more than one token as it is an offset),
find the most similar label from a list of labels whereby ahead of time you
know the positive labels and negative labels. In essence a semantic similarity
task at the token level. The difference being with these data processing
utilities they are more complicated (in the end less efficient due to batching
for negative labels) as we process the token samples at the text level all the
time rather than replicating the text per token sample which is much easier to
handle:

Each sample contains something like:

`text`: A list of token texts that represents the contextualized text. list[str].
    Example ["Did", "you", "have", "a", "good", "or", "bad", "day"]
`token_offsets`: A list of tuples which contain token start and end indexes
    for each annotation. One for each annotation. list[tuple[int, int]].
    Example [[4,5], [6,8]].
`positive_labels`: A list of positive labels for each token offset
    list[list[str]]. Example [["good%1", "great%3"], ["bad day%2"]]
`negative_labels`: A list of negative labels for each token offset, can be
    an empty list if no negative labels exist (these negative labels tend to
    be "hard" negatives as they are very close to the positive label but not quiet
    the same). Example [["ok%3"], ["good_day%4", "bad_month%3"]]
"""
