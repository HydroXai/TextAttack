import random
import re

# from textattack.transformations import WordSwap
from .word_swap import WordSwap

class WordSwapDates(WordSwap):
    """Adjusts months or years found in a sentence by a random perturbation.

    Args:
        min_month_shift (int): The minimum number of months to shift a date by.
        max_month_shift (int): The maximum number of months to shift a date by.
        min_year_shift (int): The minimum number of years to shift a date by.
        max_year_shift (int): The maximum number of years to shift a date by.
    """

    def __init__(
            self,
            min_month_shift=-12,
            max_month_shift=12,
            min_year_shift=-3,
            max_year_shift=3,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.min_month_shift = min_month_shift
            self.max_month_shift = max_month_shift
            self.min_year_shift = min_year_shift
            self.max_year_shift = max_year_shift


    def _get_replacement_words(self, word):
        replacement_words = []
        if self.max_year_shift > 0:
            m = re.match(r"\d{4}", word)
            if m: 
                year = int(word)
                year_shift = random.randint(self.min_year_shift, self.max_year_shift)
                new_year = year + year_shift
                new_word = str(new_year)
                replacement_words.append(new_word)

        months = {"january": 0, 
                  "february": 1, 
                  "march": 2, 
                  "april": 3, 
                  "may": 4, 
                  "june": 5, 
                  "july": 6, 
                  "august": 7, 
                  "september": 8, 
                  "october": 9, 
                  "november": 10, 
                  "december": 11}

        if self.max_month_shift > 0:
            new_month_cnt = months.get(word.lower())
            if new_month_cnt == None:
                return replacement_words

            new_month = self._swap_month(new_month_cnt)
            replacement_words.append(new_month)

        return replacement_words
    

    def _swap_month(self, month_cnt):
        month_shift = random.randint(self.min_month_shift, self.max_month_shift)
        new_month_cnt = (month_cnt + month_shift) % 12

        rev_months = {0: "january",
                        1: "february",
                        2: "march",
                        3: "april",
                        4: "may",
                        5: "june",
                        6: "july",
                        7: "august",
                        8: "september",
                        9: "october",
                        10: "november",
                        11: "december"}
        return rev_months.get(new_month_cnt)

