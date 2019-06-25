def make_output_line(list_of_indexes):
     
    all_letters = ["\N{HEBREW LETTER ALEF}",
    "\N{HEBREW LETTER AYIN}",
    "\N{HEBREW LETTER BET}",
    "\N{HEBREW LETTER DALET}",
    "\N{HEBREW LETTER GIMEL}",
    "\N{HEBREW LETTER HE}",
    "\N{HEBREW LETTER HET}",
    "\N{HEBREW LETTER KAF}",
    "\N{HEBREW LETTER FINAL KAF}",
    "\N{HEBREW LETTER LAMED}",
    "\N{HEBREW LETTER FINAL MEM}",
    "\N{HEBREW LETTER MEM}",
    "\N{HEBREW LETTER FINAL NUN}",
    "\N{HEBREW LETTER NUN}",
    "\N{HEBREW LETTER PE}",
    "\N{HEBREW LETTER FINAL PE}",
    "\N{HEBREW LETTER QOF}",
    "\N{HEBREW LETTER RESH}",
    "\N{HEBREW LETTER SAMEKH}",
    "\N{HEBREW LETTER SHIN}",
    "\N{HEBREW LETTER TAV}",
    "\N{HEBREW LETTER TET}",
    "\N{HEBREW LETTER FINAL TSADI}",
    "\N{HEBREW LETTER TSADI}",
    "\N{HEBREW LETTER VAV}",
    "\N{HEBREW LETTER YOD}",
    "\N{HEBREW LETTER ZAYIN}"]
    
    outputline = "" 
    for i in list_of_indexes:
        outputline += all_letters[i]
    return outputline



