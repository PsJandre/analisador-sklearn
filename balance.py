def create_partial_dataset(input_filename, output_filename, stop_condition):
    with open(input_filename, 'r', encoding='utf-8') as input_file, \
         open(output_filename, 'w', encoding='utf-8') as output_file:
        
        line_number = 1
        neg_count = 0
        pos_count = 0
        for line in input_file:
            line = line.strip()
            parts = line.split(',')
            if len(parts) == 3:
                content = parts[1]
                label = parts[2]
                
                if label == '\"pos\"' and pos_count < stop_condition:
                    output_file.write(f'"{line_number}",{content},{label}\n')
                    pos_count += 1
                    line_number += 1
                if label == '\"neg\"' and neg_count < stop_condition:
                    output_file.write(f'"{line_number}",{content},{label}\n')
                    neg_count += 1
                    line_number += 1

def count_string_occurrences(filename, target_string):
    count = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            count += line.count(target_string)
    return count

input_filename = 'output.csv'  # Nome do arquivo de entrada
output_partial_filename = 'balanced_output.csv'  # Nome do novo arquivo parcial
stop_condition = 3177  # Condição de parada

create_partial_dataset(input_filename, output_partial_filename, stop_condition)
print(f"Novo arquivo parcial criado: {output_partial_filename}")



# input_filename = 'balanced_output.csv'  # Nome do arquivo de entrada
# target_string = '\"neg\"'  # String que você deseja contar

# occurrences = count_string_occurrences(input_filename, target_string)
# print(f'A string "{target_string}" aparece {occurrences} vezes no arquivo.')





