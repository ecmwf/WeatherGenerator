from pathlib import Path
import json
import os

###########################################
class Config :

  def __init__( self) :
    pass

  def print( self) :
    self_dict = self.__dict__
    for key, value in self_dict.items() :
      if key != 'streams' :
        print("{} : {}".format( key, value))
      else :
        for rt in value :
          for k,v in rt.items() : 
            print( '{}{} : {}'.format( '' if k=='reportypes' else '  ', k, v))

  def save( self, epoch=None) :

    # save in directory with model files
    dirname = './models/{}'.format( self.run_id)
    # if not os.path.exists(dirname):
    os.makedirs( dirname, exist_ok=True)
    dirname = './models/{}'.format( self.run_id)
    # if not os.path.exists(dirname):
    os.makedirs( dirname, exist_ok=True)

    fname = './models/{}/model_{}'.format( self.run_id, self.run_id)
    epoch_str = ''
    if epoch is not None :
      epoch_str = '_latest' if epoch==-1 else '_epoch{:05d}'.format(epoch)
    fname += '{}.json'.format( epoch_str)

    json_str = json.dumps(self.__dict__ )
    with open(fname, 'w') as f :
      f.write( json_str)

  @staticmethod
  def load( run_id, epoch=None) :

    if '/' in run_id :   # assumed to be full path instead of just id
      fname = run_id
    else :
      fname = './models/{}/model_{}'.format( run_id, run_id)
      epoch_str = ''
      if epoch is not None :
        epoch_str = '_latest' if epoch==-1 else '_epoch{:05d}'.format(epoch)
      fname += '{}.json'.format( epoch_str)

    with open(fname, 'r') as f :
      json_str = f.readlines()

    cf = Config()
    cf.__dict__ = json.loads( json_str[0])

    return cf

