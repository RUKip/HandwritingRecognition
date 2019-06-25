_unicode_map = {'Alef': 0x05D0,
               'Ayin': 0x05E2,
               'Bet': 0x05D1,
               'Dalet': 0x05D3,
               'Gimel': 0x05D2,
               'He': 0x05D4,
               'Het': 0x05D7,
               'Kaf': 0x05DB,
               'Kaf-final': 0x05DA,
               'Lamed': 0x05DC,
               'Mem': 0x05DD,
               'Mem-medial': 0x05DE,
               'Nun-final': 0x05DF,
               'Nun-medial': 0x05E0,
               'Pe': 0x05E4,
               'Pe-final': 0x05E3,
               'Qof': 0x05E7,
               'Resh': 0x05E8,
               'Samekh': 0x05E1,
               'Shin': 0x05E9,
               'Taw': 0x05EA,
               'Tet': 0x05D8,
               'Tsadi-final': 0x05E5,
               'Tsadi-medial': 0x05E6,
               'Waw': 0x05D5,
               'Yod': 0x05D9,
               'Zayin': 0x05D6}


def write_results(results, filename):
    """
    Writes the text to unicode. Unicode numbers can be found here https://www.unicode.org/charts/PDF/U0590.pdf

    Args:
        results (List): Text to be written to Unicode
        filename: Name of the unicode file

    Returns:

    """
    try:
        f1 = open(filename, 'w', encoding='utf-8')
        for line in results:
            for char in reversed(line):
                f1.write(chr(_unicode_map[char]))
            f1.write('\n')
    except OSError as err:
        print('Cannot open', filename)
        print("OS error: {0}".format(err))


def main():
    filename = "testing_results.txt"
    # Some dummy text to print
    text_to_print = [['Yod', 'Shin', 'Resh', 'Alef', 'Lamed'],
                     ['Alef', 'Nun-medial', 'Shin', 'Yod'],
                     ['Ayin', 'Waw', 'Lamed', 'Mem-medial', 'Yod', 'Mem', 'Yod', 'Shin', 'Resh', 'Alef', 'Lamed', 'Alef', 'Nun-medial', 'Shin', 'Yod']]
    write_results(text_to_print, filename)


if __name__ == "__main__":
    main()

