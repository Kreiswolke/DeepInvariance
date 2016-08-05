def edit_prototxt(replacement_dict, template_file_name, new_file_name):
    """
    :param replacement_dict: {'training_data': , 'training_labels': ,
                              'testing_data': , 'testing_labels': }
    :return:
    """

    with open(new_file_name, 'w') as new_file:
        with open(template_file_name) as template_file:
            for line in template_file:
                for key in replacement_dict:
                    if key in line:
                        print(key)
                        line = line.replace(key, str(replacement_dict[key]))
                        last_key = key
                        break           
                try:
                    replacement_dict.pop(last_key, None)
                except UnboundLocalError:
                    pass
                new_file.write(line)


def edit_solver(replacement_dict_solver, template_file, new_file):
    edit_prototxt(replacement_dict_solver, template_file, new_file)