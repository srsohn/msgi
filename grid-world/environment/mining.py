import os
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
from imageio import imread

class Mining(object):
    def __init__(self):
        #map
        self.env_name = 'mining'
        nb_block = [0, 0]

        #operation
        operation_list = ['pickup','transform']

        #object
        object_list = []
        object_list.append(dict(name='workspace', pickable=False, transformable=True, oid=0, outcome=0, unique=True))
        object_list.append(dict(name='furnace', pickable=False, transformable=True, oid=1, outcome=1, unique=True))

        object_list.append(dict(name='tree', pickable=True, transformable=False, oid=2))
        object_list.append(dict(name='stone', pickable=True, transformable=False, oid=3))
        object_list.append(dict(name='grass', pickable=True, transformable=False, oid=4))
        object_list.append(dict(name='pig', pickable=True, transformable=False, oid=5))

        object_list.append(dict(name='coal', pickable=True, transformable=False, oid=6))
        object_list.append(dict(name='iron', pickable=True, transformable=False, oid=7))
        object_list.append(dict(name='silver', pickable=True, transformable=False, oid=8))
        object_list.append(dict(name='gold', pickable=True, transformable=False, oid=9))
        object_list.append(dict(imgname='diamond_ore.png', name='diamond', pickable=True, transformable=False, oid=10))

        object_list.append(dict(name='jeweler', pickable=False, transformable=True, oid=11, outcome=11, unique=True))
        object_list.append(dict(name='lumbershop', pickable=False, transformable=True, oid=12, outcome=12, unique=True))


        for i in range(len(object_list)):
            obj=object_list[i]
            if not 'imgname' in obj:
                obj['imgname'] = obj['name']+'.png'

        object_image_list = []
        img_folder = os.path.join(ROOT_DIR,'environment','config','mining')
        for obj in object_list:
            image = imread( os.path.join(img_folder,obj['imgname']) )
            object_image_list.append(image)

        self.wall_image     = imread( os.path.join(img_folder, 'mountain.png') )
        self.agent_image    = imread( os.path.join(img_folder, 'agent.png') )

        #item = agent+block+water+objects
        item_name_to_iid = dict()
        item_name_to_iid['agent']=0
        item_name_to_iid['block']=1
        item_name_to_iid['water']=2
        for obj in object_list:
            item_name_to_iid[obj['name']] = obj['oid'] + 3

        #subtask
        subtask_list = []
        subtask_list.append( dict( name='Cut wood', param=(4,2) ) )
        subtask_list.append( dict( name="Get stone",  param=(4,3) ) )
        subtask_list.append( dict( name="Get string",  param=(4,4) ) )

        subtask_list.append( dict( name="Make firewood",  param=(5,12) ) )
        subtask_list.append( dict( name="Make stick",  param=(5,12) ) )
        subtask_list.append( dict( name="Make arrow",  param=(5,12) ) )
        subtask_list.append( dict( name="Make bow",  param=(5,12) ) )

        subtask_list.append( dict( name="Make stone pickaxe",  param=(5,0) ) )
        subtask_list.append( dict( name="Hit pig", param=(4,5) ) )

        subtask_list.append( dict( name="Get coal", param=(4,6) ) )
        subtask_list.append( dict( name="Get iron ore", param=(4,7) ) )
        subtask_list.append( dict( name="Get silver ore", param=(4,8) ) )

        subtask_list.append( dict( name="Light furnace",  param=(5,1) ) )

        subtask_list.append( dict( name="Smelt iron",  param=(5,1) ) )
        subtask_list.append( dict( name="Smelt silver",  param=(5,1) ) )
        subtask_list.append( dict( name="Bake pork",  param=(5,1) ) )

        subtask_list.append( dict( name="Make iron pickaxe",  param=(5,0) ) )
        subtask_list.append( dict( name="Make silverware",  param=(5,0) ) )

        subtask_list.append( dict( name="Get gold ore",  param=(4,9) ) )
        subtask_list.append( dict( name="Get diamond ore",  param=(4,10) ) )

        subtask_list.append( dict( name="Smelt gold",  param=(5,1) ) )
        subtask_list.append( dict( name="Craft silver diamond earrings",  param=(5,11) ) )
        subtask_list.append( dict( name="Craft iron diamond rings",  param=(5,11) ) )

        subtask_list.append( dict( name="Make goldware",  param=(5,1) ) )
        subtask_list.append( dict( name="Craft gold diamond necklace",  param=(5,12) ) )
        subtask_list.append( dict( name="Make electrum bracelet",  param=(5,1) ) )

        subtask_param_to_id = dict()
        subtask_param_list = []
        for i in range(len(subtask_list)):
            subtask = subtask_list[i]
            par = subtask['param']
            subtask_param_list.append ( par )
            subtask_param_to_id[ par ] = i
        nb_obj_type = len(object_list)
        nb_operation_type = len(operation_list)

        self.operation_list=operation_list
        self.nb_operation_type=nb_operation_type

        self.object_list = object_list
        self.nb_obj_type = nb_obj_type
        self.item_name_to_iid = item_name_to_iid
        self.nb_block = nb_block
        self.subtask_list=subtask_list
        self.object_image_list=object_image_list
        self.subtask_param_list=subtask_param_list
        self.subtask_param_to_id=subtask_param_to_id

        self.nb_subtask_type = len(subtask_list)
        self.width = 10
        self.height = 10
        self.feat_dim = 4*self.nb_subtask_type+4
