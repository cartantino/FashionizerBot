from Updater import Updater
import os, sys, platform, subprocess
import matlab.engine

def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def textHandler(bot, message, chat_id, text):
	if(text == 'yes' or text == 'Yes'):
		bot.sendMessage(chat_id, "Retrieving images..")
	else:
		bot.sendMessage(chat_id, "I do not understand what you said")
	return text


def imageHandler(bot, message, chat_id, local_filename, name):
	print(local_filename)
	# send message to user
	bot.sendMessage(chat_id, "Hi, " + name + " wait until the image is ready")
	# set matlab command
	if 'Linux' in platform.system():
		matlab_cmd = '/usr/local/bin/matlab'
	else:
		matlab_cmd = '"C:\\Program Files\\MATLAB\\R2019b\\bin\\matlab.exe"'
	# set command to start matlab script "edges.m"
	cur_dir = os.path.dirname(os.path.realpath(__file__))
	cmd = matlab_cmd + " -nodesktop -nosplash -nodisplay -wait -r \"addpath(\'" + cur_dir + "\'); edges(\'" + local_filename + "\'); quit\""
	command = subprocess.Popen(cmd, shell = True)
	command.communicate()

	# send back the manipulated image
	dirName, fileBaseName, fileExtension = fileparts(local_filename)
	print("Dir = " + dirName)
	print("FileBaseName = " + fileBaseName)
	print("FileExtension = " + fileExtension)
	new_fn = os.path.join(dirName, fileBaseName + '_ok' + fileExtension)
	bot.sendImage(chat_id, new_fn, "")
	print("Image sent")
	
	bot.sendMessage(chat_id, "Would you like to retrieve most similar dresses i know?")

	

if __name__ == "__main__":
	bot_id = '1116447517:AAFIDT7Efa6-ULbi9wUZPT7lGyzm-Jxdp9s'
	updater = Updater(bot_id)
	updater.setPhotoHandler(imageHandler)
	updater.setTextHandler(textHandler)

	updater.start()