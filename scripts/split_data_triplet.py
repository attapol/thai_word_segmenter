import codecs
import random


def main():
    with codecs.open('orchid97_features.bak', encoding='utf8') as data_file:
        sequence = []
        sequence_list = []
        for line in data_file:
            character, char_type, tag = line.strip().split(' ')
            if character == 'EOS' or tag == 'O':
                if len(sequence) > 0:
                    sequence_list.append(sequence)
                sequence = []
            else:
                sequence.append((character, char_type, tag))
    random.shuffle(sequence_list)
    train_end_index = int(0.7 * len(sequence_list))
    dev_end_index = int(0.8 * len(sequence_list))
    write_to_file('train_orchid97_features.bi', sequence_list[0:train_end_index])
    write_to_file('dev_orchid97_features.bi', sequence_list[train_end_index:dev_end_index])
    write_to_file('test_orchid97_features.bi', sequence_list[dev_end_index:])

def write_to_file(file_name, sequences):
    with codecs.open(file_name, mode='w', encoding='utf8') as data_file:
        for seq in sequences:
            for line in seq:
                data_file.write(u'{} {} {}\n'.format(*line))
            data_file.write(u'\n')

if __name__ == '__main__':
    main()
