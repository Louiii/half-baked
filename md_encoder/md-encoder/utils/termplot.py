"""Taken from https://github.com/iswdp/termplot/blob/master/termplot/term_plot.py"""
from __future__ import print_function

from reprint import output


def make_col(y,max_height, plot_char):
    #requires positive value
    col_list = []
    for i in range(max_height):
        if i >= y:
            col_list.append(' ')
        else:
            col_list.append(plot_char)
    return col_list


def make_neg_col(y,max_height, plot_char):
    #requires negative value
    col_list = []
    for i in reversed(range(max_height)):
        if y >= -i:
            col_list.append(' ')
        else:
            col_list.append(plot_char)
    return col_list


def scale_data(x, plot_height):
    '''scales list data to allow for floats'''
    result = []
    z = [abs(i) for i in x]
    for i in x:
        temp = i/float(max(z))
        temp2 = temp*plot_height
        result.append(int(temp2))
    return result


def plot(x, plot_height=10, plot_char=u'\u25cf'):
    ''' takes a list of ints or floats x and makes a simple terminal histogram.
        This function will make an inaccurate plot if the length of data list is larger than the number of columns
        in the terminal.'''
    x = scale_data(x, plot_height)

    max_pos = max(x)
    max_neg = abs(min(x))

    hist_array = []
    neg_array = []

    for i in x:
        hist_array.append(make_col(i, max_pos, plot_char))

    for i in x:
        neg_array.append(make_neg_col(i, max_neg, plot_char))

    for i in range(len(neg_array)):
        neg_array[i].extend(hist_array[i])

    string = ""
    for i in reversed(range(len(neg_array[0]))):
        for j in range(len(neg_array)):
            string += neg_array[j][i]
        string += "\n"
    return string


class DisplayProgress:
    """
    with DisplayProgress() as d:
        for i in range(10):
            d.update(input_to_plot_function)
            sleep(0.5)
    """
    def __init__(self, initial_len, plot_height=10, plot_char=u'\u25cf'):
        self._reset(initial_len, plot_height)
        self.plot_char = plot_char

    def _reset(self, initial_len, plot_height):
        self.plot_height = plot_height
        self.o = output(initial_len=initial_len, interval=0)

    def __enter__(self):
        self.output_lines = self.o.__enter__()
        return self

    def update(self, x, prepend_string=None):
        string = plot(x, plot_height=self.plot_height, plot_char=self.plot_char)
        lines = string.splitlines()
        if prepend_string is not None:
            lines = [prepend_string] + lines
        if len(lines) > len(self.output_lines):
            self._reset(len(lines) + 5, self.plot_height)
            self.__enter__()
            return self.update(x)

        for i, line in enumerate(lines):
            self.output_lines[i] = line

    def __exit__(self, *args):
        print()
