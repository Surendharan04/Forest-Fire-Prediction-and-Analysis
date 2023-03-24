from email.policy import default
from unittest.util import _MAX_LENGTH
from django.db import models

# Create your models here.
class UserPredictDataModel(models.Model):
    Temperature = models.CharField(max_length=100)
    RH = models.CharField(max_length=100)
    Ws = models.CharField(max_length=100)
    Rain = models.CharField(max_length=100)
    FFMC = models.CharField(max_length=100)
    DMC = models.CharField(max_length=100)
    DC = models.CharField(max_length=100)
    ISI = models.CharField(max_length=100)
    BUI = models.CharField(max_length=100)
    FWI = models.CharField(max_length=100)
    Classes = models.CharField(max_length=100)


def __str__(self):
    return self.Temperature, self.RH,self.Ws,self.Rain,self.FFMC,self.DMC,self.DC,self.ISI,self.BUI,self.FWI,self.Classes

class FeedModel(models.Model):
    Feedback = models.TextField(max_length=100)

    def __str__(self):
        return str(self.Feedback)