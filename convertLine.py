def modify_lines(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as input_file, open(output_filename, 'w', encoding='utf-8') as output_file:
        line_number = 1
        for line in input_file:
            line = line.strip()
            parts = line.rsplit(";", 1)
            if len(parts) != 2 or not parts[1].isdigit() or parts[1] == "3":
                continue  # Ignora linhas sem número no final ou com número igual a 3
            content, number = parts
            content = content.strip()  # Remove espaços em branco no início e final da frase
            modified_line = f'"{line_number}","{content}","{number}"\n'
            output_file.write(modified_line)
            line_number += 1

input_filename = 'base.csv'  # Coloque o nome do seu arquivo de entrada aqui
output_filename = 'output.csv'  # Escolha um nome para o arquivo de saída

modify_lines(input_filename, output_filename)
print("Linhas modificadas e escritas no arquivo de saída.")