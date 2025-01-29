
import code
import argparse
import dictdiffer

from obslearn.utils.config import Config

if __name__ == '__main__' :

  parser = argparse.ArgumentParser( )
  parser.add_argument( '-r1', '--run_id_1', required=True)
  parser.add_argument( '-r2', '--run_id_2', required=True)
  args = parser.parse_args()

  cf1 = Config.load( args.run_id_1)
  cf2 = Config.load( args.run_id_2)
  # print(cf1.__dict__)
  result = dictdiffer.diff( cf1.__dict__, cf2.__dict__)
  for item in list(result) :

    # TODO: if streams_directory differs than we need to manually compare streams using name
    # since index-based comparison by dictdiffer is meaningless

    # # for streams, translate index in list of streams to stream name   
    # if item[1][0] == 'streams' :
    #   name = cf1.streams[item[1][1]]['name']
    #   item[1][1] = name 

    print( f'{item[1]} :: {item[2]}')
