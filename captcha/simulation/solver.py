import os
import cv2
import numpy as np
import imageprocessor as proc


def rename_to_prediction(image_path, model, labels, new_size):
    letters = proc.extract_letters(image_path, expected_char_num=-1)
    if len(letters) > 0:
        for ind, letter in enumerate(letters):
            letters[ind] = cv2.resize(letter, new_size, interpolation=cv2.INTER_LINEAR)
        w, h = new_size
        predicted_word = ""
        for prediction in model.predict(np.reshape(letters, [-1, w, h, 1])):
            predicted_word += labels[np.argmax(prediction).item()]
        dirname = os.path.dirname(image_path)
        extension = os.path.splitext(os.path.basename(image_path))[1]
        os.rename(image_path, os.path.join(dirname, predicted_word + extension))
    else:
        print("Image is not contain any recognized contour.")


def do_validation(dataset_dir, model, labels, new_size, batch_size=64):
    counter = {"incorrect": 0, "correct": 0, "captcha": 0}
    valid_captcha_num = 0
    file_extension = ""

    def process_batch():
        words = ""
        for word in valid_batch:
            words += word
        valid_chars = list(words)

        w, h = new_size
        predicted_letters = []
        for prediction in model.predict(np.reshape(letter_batch, [-1, w, h, 1])):
            predicted_letters.append(labels[np.argmax(prediction).item()])

        incorrect_num = counter["incorrect"]
        correct_num = counter["correct"]
        captcha_num = counter["captcha"]
        solved_batch = []
        predicted_word = ""
        missed_dir = "missed"
        for ind, (valid_char, predicted_letter) in enumerate(zip(valid_chars, predicted_letters)):
            predicted_word += predicted_letter
            if valid_char != predicted_letter:
                incorrect_num += 1

            if (ind + 1) % word_length == 0:
                solved_batch.append([valid_batch[ind // word_length], predicted_word])
                predicted_word = ""

        for valid_word, predicted_word in solved_batch:
            captcha_num += 1
            if valid_word == predicted_word:
                print("#{}: Captcha {} solved.".format(captcha_num, valid_word))
                correct_num += 1
            else:
                print("#{}: Captcha {} has incorrect prediction ({}).".format(captcha_num, valid_word, predicted_word))
                os.path.exists(missed_dir) or os.makedirs(missed_dir)
                os.system("cp {} {}".format(os.path.join(os.path.abspath(dataset_dir), valid_word + file_extension),
                                            os.path.join(os.path.abspath(missed_dir), predicted_word + file_extension)))

        counter["incorrect"] = incorrect_num
        counter["correct"] = correct_num
        counter["captcha"] = captcha_num

    def print_statistics():
        incorrect_letter_num = counter["incorrect"]
        correct_captcha_num = counter["correct"]
        print()
        print("Statistics:")
        print("Number of mispredicted letters: {}".format(incorrect_letter_num))
        letter_num = valid_captcha_num * word_length
        matched_num = letter_num - incorrect_letter_num
        print("Correct letter/All letter: {}/{} ({:.2f}%)".format(matched_num, letter_num, 100 * matched_num / letter_num))
        print("Correct captcha/All captcha: {}/{} ({:.2f}%)".format(correct_captcha_num, valid_captcha_num, 100 * correct_captcha_num / valid_captcha_num))

    letter_batch = []
    valid_batch = []
    word_length = 0
    for captcha in os.listdir(dataset_dir):
        captcha_word, file_extension = os.path.splitext(captcha)
        word_length = len(captcha_word)
        letters = proc.extract_letters(os.path.join(dataset_dir, captcha))
        if letters is not None:
            for letter in letters:
                letter = cv2.resize(letter, new_size, interpolation=cv2.INTER_LINEAR)
                letter_batch.append(letter)
            valid_batch.append(captcha_word)
            valid_captcha_num += 1

            if len(valid_batch) == batch_size:
                process_batch()
                letter_batch = []
                valid_batch = []

    if len(valid_batch) > 0:
        process_batch()

    print_statistics()
