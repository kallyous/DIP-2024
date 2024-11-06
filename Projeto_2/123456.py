import argparse

### START CODE HERE ###
def util_function_1 ():

    dummy_value = 3.0

    return dummy_value
### END CODE HERE ###

def main(registration_number, input_filename):
    """
    Main function to calculate the result based on input parameters.
    
    Args:
        registration_number (str): Student's registration number.
        input_filename (str): Absolute input file name as; <PATH + IMAGEFILE + EXTENSION> eg. 'C://User//Student//image.png' OR 'C://User//Student//image.jpg'.
                              Absolute input file name as; <PATH +  DATASET  + EXTENSION> eg. 'C://User//Student//dataset.csv' OR 'C://User//Student//dataset.npy'
    """
    
    ### START CODE HERE ###
    value_1 = 2.0
    value_2 = util_function_1()
    output_value = value_1 + value_2
    ### END CODE HERE ###
   
    return output_value

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Student project script template")
    
    # Add arguments
    parser.add_argument('--registration_number', type=str, required=True, help="Student's registration number")
    parser.add_argument('--input_filename', type=str, required=True, help="Absolute input file name")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    output_value = main(args.registration_number, args.input_filename)

    # # Write the result to file
    with open(args.registration_number + '.csv', 'w') as f:
        f.write(str(output_value))