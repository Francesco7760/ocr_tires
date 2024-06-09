import configparser as cp

config = cp.ConfigParser()
config.read(r'config.ini')

print(str(config.get('directory', 'binary_directory_path')))

