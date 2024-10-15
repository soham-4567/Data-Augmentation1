
# import json
#
# # Load config from JSON file
# with open('add_config.json', 'r') as file:
#     config = json.load(file)
#
# # Get operation and parameters from the config
# operation = config["operation"]
# parameters = config["parameters"]
#
# # Input values for the parameters
# values = {}
# for param in parameters:
#     values[param] = float(input(f"Enter value for {param}: "))
#
# # Use eval to dynamically perform the operation from the config
# result = eval(operation, values)
#
# # Display the result
# print(f"ResultJSON: {result}")
#
#
#
# # Load config from TXT file
# config = {}
# with open('add_config.txt', 'r') as file:
#     for line in file:
#         key, value = line.strip().split(' = ')
#         config[key] = value
#
# # Get operation and parameters
# operation = config["operation"]
# parameters = config["parameters"].split(", ")
#
# # Input values for the parameters
# values = {}
# for param in parameters:
#     values[param] = float(input(f"Enter value for {param}: "))
#
# # Perform the operation using eval
# result = eval(operation, values)
#
# # Display the result
# print(f"ResultTXT: {result}")
#
#
# # Load config from OUT file
# config = {}
# with open('add_config.out', 'r') as file:
#     for line in file:
#         key, value = line.strip().split(' = ')
#         config[key] = value
#
# # Get operation and parameters
# operation = config["operation"]
# parameters = config["parameters"].split(", ")
#
# # Input values for the parameters
# values = {}
# for param in parameters:
#     values[param] = float(input(f"Enter value for {param}: "))
#
# # Perform the operation using eval
# result = eval(operation, values)
#
# # Display the result
# print(f"ResultOUT: {result}")
#
#
# import configparser
#
# # Load config from INI file
# config = configparser.ConfigParser()
# config.read('add_config.ini')
#
# # Get operation and parameters from the INI file
# operation = config["operation"]["expression"]
# parameters = config["parameters"]["list"].split(", ")
#
# # Input values for the parameters
# values = {}
# for param in parameters:
#     values[param] = float(input(f"Enter value for {param}: "))
#
# # Perform the operation using eval
# result = eval(operation, values)
#
# # Display the result
# print(f"ResultINI: {result}")

import json

# Load config from JSON file
with open('add_config.json', 'r') as file:
    config = json.load(file)

# Get numbers
a = config["numbers"]["a"]
b = config["numbers"]["b"]

# Perform addition
result = a + b
print(f"ResultJSON: {result}")

import configparser

# Load config from INI file
config = configparser.ConfigParser()
config.read('add_config.ini')

# Get numbers
a = int(config["numbers"]["a"])
b = int(config["numbers"]["b"])

# Perform addition
result = a + b
print(f"ResultINI: {result}")

# Load config from TXT file
config = {}
with open('add_config.txt', 'r') as file:
    for line in file:
        name, value = line.split('=')
        config[name.strip()] = int(value.strip())

# Get numbers
a = config["a"]
b = config["b"]

# Perform addition
result = a + b
print(f"ResultTXT: {result}")

# Load config from OUT file
config = {}
with open('add_config.out', 'r') as file:
    for line in file:
        name, value = line.split('=')
        config[name.strip()] = int(value.strip())

# Get numbers
a = config["a"]
b = config["b"]

# Perform addition
result = a + b
print(f"ResultOUT: {result}")


