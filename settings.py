import data_generate_pickle
import config
import pickle
import logging
import traceback


def reset_log():
    with open('info.log', 'w'):
        pass
    with open('settings.log', 'w'):
        pass


def main():
        # run for generate pickle file
    logging.info('called')
    data_generate_pickle.DataPrepPickle()


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='settings.log', filemode='a',
                        level=logging.DEBUG, format=FORMAT)
    try:
        main()
    except Exception as e:
        traceback.print_exc()  # console print exception
        logging.exception("Exception occurred")  # log exception
