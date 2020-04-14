import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Playground(object):
    def __init__(self):
        #map
        self.env_name = 'playground'
        nb_block = [0, 0]

        #operation
        operation_list = ['pickup','transform']

        #object
        object_list = []
        object_list.append(dict(imgname='cow.png', name='cow', pickable=True, transformable=True, oid=0, outcome=8, updateable=True, speed=0.1))
        object_list.append(dict(imgname='duck.png', name='duck', pickable=True, transformable=True, oid=1, outcome=8, updateable=True, speed=0.2))
        object_list.append(dict(imgname='milk.png', name='milk', pickable=True, transformable=True, oid=2, outcome=8, updateable=False, speed=0))
        object_list.append(dict(imgname='chest.png', name='chest', pickable=True, transformable=True, oid=3, outcome=8, updateable=False, speed=0))
        object_list.append(dict(imgname='diamond.png', name='diamond', pickable=True, transformable=True, oid=4, outcome=8, updateable=False, speed=0))
        object_list.append(dict(imgname='steak.png', name='steak', pickable=True, transformable=True, oid=5, outcome=8, updateable=False, speed=0))
        object_list.append(dict(imgname='egg.png', name='egg', pickable=True, transformable=True, oid=6, outcome=8, updateable=False, speed=0))
        object_list.append(dict(imgname='heart.png', name='heart', pickable=True, transformable=True, oid=7, outcome=8, updateable=False, speed=0))
        object_list.append(dict(imgname='ice.png', name='ice', pickable=False, transformable=False, oid=8, outcome=8, updateable=False, speed=0))

        #item = agent+block+water+objects
        item_name_to_iid = dict()
        item_name_to_iid['agent']=0
        item_name_to_iid['block']=1
        item_name_to_iid['water']=2
        for obj in object_list:
            item_name_to_iid[obj['name']] = obj['oid'] + 3

        #subtask
        subtask_list = []
        subtask_param_to_id = dict()
        subtask_param_list = []
        for i in range(len(operation_list)):
            oper = operation_list[i]
            for j in range(len(object_list)):
                obj = object_list[j]
                if (i==0 and obj['pickable']) or (i==1 and obj['transformable']):
                    item = dict( param=(i+4,j), oper=oper, obj=obj )

                    subtask_list.append( item )
                    subtask_param_list.append ( (i+4,j) )
                    subtask_param_to_id[ (i+4,j) ] = len(subtask_list)-1
        nb_obj_type = len(object_list)
        nb_operation_type = len(operation_list)

        self.operation_list=operation_list
        self.nb_operation_type=nb_operation_type

        self.object_list = object_list
        self.nb_obj_type = nb_obj_type # nb_channel in observation tensor.
        self.item_name_to_iid = item_name_to_iid
        self.nb_block = nb_block
        self.subtask_list=subtask_list
        self.subtask_param_list=subtask_param_list
        self.subtask_param_to_id=subtask_param_to_id

        self.nb_subtask_type = len(subtask_list) #16
        self.width = 10
        self.height = 10
        self.feat_dim = 4*self.nb_subtask_type+4
