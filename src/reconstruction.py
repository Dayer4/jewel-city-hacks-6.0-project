
def reconstruct_text(recognized_lines):

    reconstructed_lines = []

    for line in recognized_lines:

        text = ""

        for character in line:

            # Insert spaces between words
            if character["space_before"] and text:
                text += " "

            text += character["text"]

        reconstructed_lines.append(text)

    return "\n".join(reconstructed_lines)
