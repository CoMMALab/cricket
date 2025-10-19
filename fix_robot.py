import re
import sys



def read_file(filepath):
    with open(filepath, 'r') as f:
        file_content = f.read()
    return file_content

def write_file(filepath, file_content):
    with open(filepath, 'w') as f:
        # file_content = f.read()
        f.write(file_content)


def fix_conditionals(file_content):


    find_replace_pairs = [
        (r"if\s*\(\s*(.*?)\s*>=\s*(.*?)\s*\)\s*\{\s*\n\s*(\w+(?:\[\d+\])?)\s*=\s*(.*?);\s*\}\s*else\s*\{\s*\3\s*=\s*(.*?);\s*\}", r'\3 = blend(\4, \5, \1 - \2);'),
        (r"if\s*\(\s*(.*?)\s*>\s*(.*?)\s*\)\s*\{\s*\n\s*(\w+(?:\[\d+\])?)\s*=\s*(.*?);\s*\}\s*else\s*\{\s*\3\s*=\s*(.*?);\s*\}", r'\3 = blend(\5, \4, \2 - \1);'),
        (r"if\s*\(\s*(.*?)\s*<=\s*(.*?)\s*\)\s*\{\s*\n\s*(\w+(?:\[\d+\])?)\s*=\s*(.*?);\s*\}\s*else\s*\{\s*\3\s*=\s*(.*?);\s*\}", r'\3 = blend(\4, \5, \2 - \1);'),
        (r"if\s*\(\s*(.*?)\s*<\s*(.*?)\s*\)\s*\{\s*\n\s*(\w+(?:\[\d+\])?)\s*=\s*(.*?);\s*\}\s*else\s*\{\s*\3\s*=\s*(.*?);\s*\}", r'\3 = blend(\5, \4, \1 - \2);'),
    ]

    for find, replace in find_replace_pairs:
        matches = re.findall(find, file_content)
        print(find, matches)
        file_content = re.sub(find, replace, file_content)

    # print(file_content)
    return file_content


def main():
    file_path = sys.argv[1]
    file_content = read_file(file_path)
    replaced_file_content = fix_conditionals(file_content)
    write_file(file_path, replaced_file_content)

if __name__ == '__main__':
    main()
