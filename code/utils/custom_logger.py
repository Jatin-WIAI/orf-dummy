import logging

class CustomLogger(logging.Logger):
    def __init__(self, name, filename=None):
        super().__init__(name)
        self.filename = filename
        self.set_console_handler()
        if filename:
            self.set_file_handler()
            

    def set_console_handler(self):
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(c_format)
        self.addHandler(c_handler)

    def set_file_handler(self):
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler = logging.FileHandler(self.filename)
        f_handler.setFormatter(f_format)
        self.addHandler(f_handler)