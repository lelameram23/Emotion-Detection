# Yazım Düzeltme
import pkg_resources
from symspellpy import SymSpell, Verbosity

class Spell_Checker():
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
        self.bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")

        self.sym_spell.load_dictionary(self.dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(self.bigram_path, term_index=0, count_index=2)

    def Correct_It(self, data):
        suggestions = self.sym_spell.lookup_compound(data, max_edit_distance=2,
                                            transfer_casing=True)

        clean_data = list()
        for suggestion in suggestions:
            clean_data.append(str(suggestion.term))

        correct_data = " ".join(clean_data)

        return correct_data

