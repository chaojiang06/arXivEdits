
import diff_match_patch as dmp_module
import argparse
import json
import copy


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def char_offsets_from_diff_v1(diff):
    """Find character offsets for each diff."""
    char_offset1, char_offset2 = 0, 0
    char_mappings = []

    for operation, segment in diff:
        segment_length = len(segment)
        if operation == 0:  # No change
            char_offset1 += segment_length
            char_offset2 += segment_length
        elif operation == -1:  # Deletion from sent1
            char_mappings.append(
                ('deletion', char_offset1, char_offset1 + segment_length))
            char_offset1 += segment_length
        elif operation == 1:  # Insertion in sent2
            char_mappings.append(
                ('insertion', char_offset2, char_offset2 + segment_length))
            char_offset2 += segment_length

    return char_mappings


def char_offsets_from_diff_v2(diff):
    """Find character offsets for each diff."""
    char_offset1, char_offset2 = 0, 0
    char_mappings = []

    for operation, segment in diff:
        segment_length = len(segment)
        if operation == 0:  # No change

            char_mappings.append((
                'keep', (char_offset1, char_offset1 + segment_length),
                (char_offset2, char_offset2 + segment_length))
            )

            char_offset1 += segment_length
            char_offset2 += segment_length

        elif operation == -1:  # Deletion from sent1
            char_mappings.append(
                ('deletion', (char_offset1, char_offset1 + segment_length), (None, None)))
            char_offset1 += segment_length
        elif operation == 1:  # Insertion in sent2
            char_mappings.append(
                ('insertion', (None, None), (char_offset2, char_offset2 + segment_length)))
            char_offset2 += segment_length

    return char_mappings


def char_offset_to_token_offset(sentence, start_char, end_char):
    """Convert a character offset range to a token offset range."""
    # tokens = sentence.split()
    # start_token = None
    # end_token = None

    # for i, token in enumerate(tokens):

    #     char_count += len(token)
    #     if start_char <= char_count and start_token is None:
    #         start_token = i
    #     if end_char <= char_count:
    #         end_token = i
    #         break
    #     char_count += 1

    start_token = len(sentence[:start_char].strip().rstrip().split())
    end_token = len(sentence[:end_char].strip().rstrip().split())
    # print("AAA", start_token, end_token)
    # print("BBB", sentence, start_char, end_char)
    return start_token, end_token


def generate_human_readable_diff_v20(sent1, sent2, diff):
    """Generate a human-readable diff where adjacent deletions and insertions are merged as substitutions."""

    char_mappings = char_offsets_from_diff_v1(diff)
    output = []
    idx = 0

    while idx < len(char_mappings):
        operation, start_char, end_char = char_mappings[idx]

        if operation == 'deletion':
            token_start1, token_end1 = char_offset_to_token_offset(
                sent1, start_char, end_char)

            # Check if the next operation is an insertion
            if idx + 1 < len(char_mappings) and char_mappings[idx + 1][0] == 'insertion':
                next_operation, next_start_char, next_end_char = char_mappings[idx + 1]
                token_start2, token_end2 = char_offset_to_token_offset(
                    sent2, next_start_char, next_end_char)
                output.append(
                    ["Substitute", (token_start1, token_end1), (token_start2, token_end2)])
                idx += 1  # Skip the next mapping since we've already processed it
            else:
                output.append(
                    ["Deletion", (token_start1, token_end1), (None, None)])

        elif operation == 'insertion':
            token_start2, token_end2 = char_offset_to_token_offset(
                sent2, start_char, end_char)
            output.append(["Insertion", (None, None),
                          (token_start2, token_end2)])

        idx += 1

    return output


def handle_inword_operation(diff, text1, text2):

    char_mappings = char_offsets_from_diff_v2(diff)
    processed_diff = []

    # print(char_mappings)
    # print(diff)
    skip_next = False

    for idx, i in enumerate(char_mappings):
        # print(idx, i)
        FLAG = False

        if skip_next == True:
            skip_next = False
            continue

        j = diff[idx]
        (operation, (start_char1, end_char1), (start_char2, end_char2)) = i

        if operation == 'deletion':
            start_char, end_char = start_char1, end_char1
            token_start, token_end = char_offset_to_token_offset(
                text1, start_char, end_char)

            # check if lonely deletion
            if operation == 'deletion' and ((idx + 1 < len(char_mappings) and char_mappings[idx + 1][0] == 'keep') or (idx == len(char_mappings) - 1)) and ((idx - 1 >= 0 and char_mappings[idx - 1][0] == 'keep') or (idx == 0)):

                # check if a in-word operation
                if " ".join(text1.split()[token_start: token_end]) != text1[start_char: end_char].strip().rstrip():
                    # handle in-word operation
                    # print("in-word deletion")
                    token_start -= 1
                    # print(token_start, token_end)
                    # print(start_char, end_char)
                    # print(" ".join(text1.split()[token_start: token_end]))
                    # print(text1[start_char: end_char])

                    all_start = len(" ".join(text1.split()[:token_start])) + 1
                    all_end = len(" ".join(text1.split()[:token_end]))
                    # print(text1[all_start: start_char] + text1[end_char:all_end])
                    if idx - 1 >= 0:
                        processed_diff[-1] = (0, diff[idx - 1][1]
                                              [: -len(text1[all_start: start_char])])
                    processed_diff.append(
                        list([-1, " ".join(text1.split()[token_start: token_end])]))
                    processed_diff.append(
                        list([1, text1[all_start: start_char] + text1[end_char:all_end]]))
                    if idx + 1 < len(char_mappings):
                        diff[idx + 1] = (0, diff[idx + 1][1]
                                         [len(text1[end_char:all_end]):])

                    FLAG = True

            # check if in-word substitution

            if (operation == 'deletion') and (idx + 1 < len(char_mappings)) and (char_mappings[idx + 1][0] == 'insertion') and ((idx == 0) or (idx - 1 >= 0 and char_mappings[idx - 1][0] == 'keep')) and ((idx + 2 == len(char_mappings)) or (idx + 2 < len(char_mappings) and char_mappings[idx + 2][0] == 'keep')):

                # check if a in-word operation
                if " ".join(text1.split()[token_start: token_end]) != text1[start_char: end_char].strip().rstrip():
                    # handle in-word operation
                    # print("in-word substitution")

                    # get the insertion, this is char offsets for the text2
                    next_start_char, next_end_char = char_mappings[idx + 1][2]

                    '''
                    this is the logic
                    if the previous keep string exists, and the last char is not " ", I want to process it
                    if the next keep string exists, and the first char is not " ", I want to process it
                    '''

                    prepend = ""
                    if (idx - 1 >= 0) and (char_mappings[idx - 1][0] == 'keep') and (text1[char_mappings[idx - 1][1][1] - 1] != " "):
                        prepend = text1[char_mappings[idx - 1][1][0]
                            : char_mappings[idx - 1][1][1]].split()[-1]
                        # print("CCC", processed_diff[-1][1])
                        processed_diff[-1][1] = processed_diff[-1][1][:-
                                                                      len(prepend)]

                    append = ""
                    if (idx + 2 < len(char_mappings)) and (char_mappings[idx + 2][0] == 'keep') and (text1[char_mappings[idx + 2][1][0]] != " "):
                        append = text1[char_mappings[idx + 2][1][0]
                            : char_mappings[idx + 2][1][1]].split()[0]
                        diff[idx + 2] = (0, diff[idx + 2][1][len(append):])

                    processed_diff.append(
                        list([-1, prepend + text1[char_mappings[idx][1][0]: char_mappings[idx][1][1]] + append]))
                    processed_diff.append(
                        list([1, prepend + text2[next_start_char: next_end_char] + append]))

                    skip_next = True
                    FLAG = True

        if operation == 'insertion':
            start_char, end_char = start_char2, end_char2
            token_start, token_end = char_offset_to_token_offset(
                text2, start_char, end_char)

            # check if lonely insertion
            if operation == 'insertion' and ((idx + 1 < len(char_mappings) and char_mappings[idx + 1][0] == 'keep') or (idx == len(char_mappings) - 1)) and ((idx - 1 >= 0 and char_mappings[idx - 1][0] == 'keep') or (idx == 0)):

                # check if a in-word operation
                if " ".join(text2.split()[token_start: token_end]) != text2[start_char: end_char].strip().rstrip():
                    # handle in-word operation
                    # print("in-word insertion")
                    # print(start_char, end_char)
                    token_start -= 1
                    # print(token_start, token_end)
                    # print(" ".join(text2.split()[token_start: token_end]))
                    # print(text2[start_char: end_char].strip().rstrip())

                    all_start = len(" ".join(text2.split()[:token_start])) + 1
                    all_end = len(" ".join(text2.split()[:token_end]))
                    # print(text2[all_start: start_char] + text2[end_char:all_end])

                    if idx - 1 >= 0:
                        # print(processed_diff[-1])
                        # print(diff[idx - 1])
                        # print(text2)
                        processed_diff[-1] = (0, diff[idx - 1][1]
                                              [: -len(text2[all_start: start_char])])

                    processed_diff.append(
                        list([-1, text2[all_start: start_char] + text2[end_char:all_end]]))
                    processed_diff.append(
                        list([1, " ".join(text2.split()[token_start: token_end])]))

                    if idx + 1 < len(char_mappings):
                        diff[idx + 1] = (0, diff[idx + 1][1]
                                         [len(text2[end_char:all_end]):])

                    FLAG = True

        # print(FLAG, i, j)
        if FLAG == False:
            processed_diff.append(list(j))
    return processed_diff


if __name__ == "__main__":

    # below is a minimal test case

    # # text1 = "Hello World . I am chao . The world is beautiful . It is nice ."
    # # text2 = "Hello . I am Sam . The world is very big . It is nice . The universe is even bigger ."

    # text1 = "This clearly is an oversimplification , as the moment-zero map shows obvious evidence of non-exponential emission ( see Figure [REF] ) ."
    # text2 = "This clearly is an oversimplification , as the moment 0 map shows obvious evidence of nonexponential emission ( see Figure [REF] ) ."

    # # text1 = "The algorithm is implemented in a multi-threaded fashion using C++11 and GNU Scientific Library ."
    # # text2 = "The algorithm is implemented in a multi-threaded fashion using C++11 and GNU Scientific Library with a shared memory architecture ."
    # dmp = dmp_module.diff_match_patch()
    # diff = dmp.diff_main(text1, text2)
    # dmp.diff_cleanupSemantic(diff)

    # processed_diff = handle_inword_operation(copy.deepcopy(diff), text1, text2)
    # print(generate_human_readable_diff_v20(text1, text2, processed_diff))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    data = read_json(args.input)
    for k, v in data.items():
        text1 = " ".join(v['sentence-1'].split())
        text2 = " ".join(v['sentence-2'].split())

        if v['sentence-1'] != text1:
            print("Please only use whitespace as the separator.")
            print("before: ", v['sentence-1'])
            print("after: ", text1)
            print("-----")

        if v['sentence-2'] != text2:
            print("Please only use whitespace as the separator.")
            print("before: ", v['sentence-2'])
            print("after: ", text2)
            print("-----")

        dmp = dmp_module.diff_match_patch()
        diff = dmp.diff_main(text1, text2)
        dmp.diff_cleanupSemantic(diff)

        processed_diff = handle_inword_operation(
            copy.deepcopy(diff), text1, text2)
        output = generate_human_readable_diff_v20(text1, text2, processed_diff)
        # print(output)

        tmp = {}
        for idx_i, i in enumerate(output):
            if i[0] == "Substitute":
                tmp[idx_i] = {
                    "type": i[0],
                    "intention": "Content",
                    "sentence-1-token-indices": [i[1][0], i[1][1]],
                    "sentence-2-token-indices": [i[2][0], i[2][1]]
                }
            elif i[0] == "Insertion":
                tmp[idx_i] = {
                    "type": i[0],
                    "intention": "Content",
                    "sentence-1-token-indices": None,
                    "sentence-2-token-indices": [i[2][0], i[2][1]]
                }
            elif i[0] == "Deletion":
                tmp[idx_i] = {
                    "type": i[0],
                    "intention": "Content",
                    "sentence-1-token-indices": [i[1][0], i[1][1]],
                    "sentence-2-token-indices": None
                }

        data[k]["edits-combination-0"] = tmp

    write_json(data, args.output)
