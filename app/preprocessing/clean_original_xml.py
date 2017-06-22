"""
This submodule cleans up the original XML file we got from M&S.
"""

import codecs

def clean_original_xml(data_folder, original_file, output_file):
    """
    The data we got from Richard has an XML definition tag in every second
    line, thus making the file not a valid XML file. This  script removes them.

    :param data_folder: [string], absolute path to data folder
    :param original_file: [string], name of original XML file to clean
    :param output_file: [string], name of output file we will create
    :return: None
    """

    # open files
    original_file = codecs.open(data_folder + original_file, 'rU', 'utf-8')
    output_file = codecs.open(data_folder + output_file, 'w', 'utf-8')

    # write XML definition into output file
    output_file .write('<?xml version="1.0" encoding="utf-8"?>\n<root>\n')
    for i in original_file:
        # remove unnecessary XML definitions
        if i[:4] != '<?xm':
            output_file.write(i)
    output_file.write('</root>')
    output_file.close()
    original_file.close()