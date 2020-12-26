import smtplib
from keras.callbacks import *
from keras import backend as K
from statistics import mean

carriers = {
	'att':    '@mms.att.net',
	'tmobile':' @tmomail.net',
	'verizon':  '@vtext.com',
	'sprint':   '@page.nextel.com'
}

class SMS(Callback):
    def __init__(self, phonenumber):
        self.phonenumber = phonenumber

    def send(self, message):
        # Replace the number with your own, or consider using an argument\dict for multiple people.
        to_number = '{}{}'.format(self.phonenumber, carriers['verizon'])
        auth = ('feyisope.reginald.eweje@gmail.com', '467E1ade!')
        
        # Establish a secure session with gmail's outgoing SMTP server using your gmail account
        server = smtplib.SMTP( "smtp.gmail.com", 587 )
        server.starttls()
        server.ehlo()
        server.login(auth[0], auth[1])
        
        # Send text message through SMS gateway of destination number
        server.sendmail( auth[0], to_number, message)
    
    def on_train_begin(self, logs=None):
        self.send("Training is underway!")
        
    def on_epoch_end(self, epoch, logs=None):
        actual_epoch = epoch+1
        history = self.model.history.history
        if actual_epoch%10==0:
            self.send("Just got to epoch {}. Avg val_acc for the last 10 epochs = {}.".format(actual_epoch, str(round(mean(history['val_accuracy'][-10:]),3))))
