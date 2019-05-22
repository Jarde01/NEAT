import configparser


class Config:
    config = configparser.ConfigParser()
    config.read("config.ini")
