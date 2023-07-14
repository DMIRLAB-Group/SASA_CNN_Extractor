

def get_dataset_config_class(dataset_name):
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class Boiler:
    def __init__(self):
        super(Boiler, self).__init__()
        self.data_base_path = './datasets/Boiler'
        self.scenarios = [("1", "2"), ("1", "3"), ("2", "1"), ("2", "3"), ("3", "1"), ('3', '2')]
        self.input_dim = 15
        self.class_num = 2
        self.window_size = 6
        self.time_interval = 16
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.data_base_path = './datasets/HAR'
        self.window_size = 100
        self.input_dim = 9
        self.class_num = 6
        self.time_interval = 16
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)
        self.scenarios = [("10", "16"), ("10", "18"), ("11", "18"), ("12", "16"), ("12", "18")] # adatime
        # self.scenarios = [('27','16'),('12','18'),('7','16'),('17','16'),('28','21'),('28','16'),('20','18'),('9','16'),('13','18'),('29','18'),('11','18'),('12','16'),('21','16'),('5','16'),('25','18'),('4','18'),('17','18'),('27','18'),('15','18'),('10','18'),('6','18'),('30','18'),('24','18'),('5','18'),('27','21'),('18','24'),('13','16'),('18','16'),('18','21'),('26','16'),('26','18'),('24','16'),('14','18'),('10','16'),('21','18'),('7','18'),('17','24'),('29','16'),('17','21'),('6','16'),('12','24'),('20','16'),('7','21'),('28','27')]
        # self.scenarios = [('12', '16'), ('9', '18'), ('3', '16'), ('21', '5'), ('28', '9')]  # 嘉文师姐
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 100
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 100
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 100
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 100
        self.prej_input_channels = 1


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.data_base_path = './datasets/EEG'
        self.window_size = 6
        self.input_dim = 15
        self.class_num = 5
        self.time_interval = 16
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)


        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
                          ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]
        # self.scenarios = [('17', '8'), ('15', '8'), ('15', '4'), ('1', '8'), ('9', '8'), ('17', '4'), ('1', '2'),
        #                   ('10', '4'), ('14', '4'), ('19', '8'), ('10', '8'), ('1', '4'), ('17', '2'), ('4', '8'),
        #                   ('12', '4'), ('12', '8'), ('11', '4'), ('19', '4'), ('18', '8'), ('12', '2'), ('14', '2'),
        #                   ('18', '4'), ('9', '2'), ('7', '2'), ('14', '8'), ('17', '10'), ('16', '4'), ('16', '2'),
        #                   ('8', '4'), ('7', '4'), ('9', '1'), ('3', '4'), ('8', '2'), ('18', '2'), ('16', '6'),
        #                   ('7', '8'), ('11', '8'), ('14', '10'), ('9', '6'),
        #                   ]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 100
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32, 64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15  # 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 100
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()

        self.data_base_path = './datasets/WISDM'
        self.window_size = 100  # 100 128
        self.input_dim = 3
        self.class_num = 6
        self.time_interval = 16
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)


        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 100
        # self.scenarios = [('30','7'),('28','7'),('13','7'),('5','7'),('14','7'),('4','8'),('34','10'),('2','7'),('13','10'),('7','30'),('5','10'),('21','7'),('13','2'),('10','1'),('17','10'),('33','7'),('2','10'),('7','2'),('7','10'),('1','8'),('27','8'),('32','8'),('8','1'),('26','1'),('4','1'),('2','1'),('10','8'),('18','7'),('34','7'),('8','10'),('26','10'),('35','10'),('12','10'),('18','2'),('14','10'),('28','10'),('19','7'),('33','10'),('14','30'),('18','10'),('33','1'),('4','10'),('1','10')]
        self.scenarios = [("1", "8"), ("1", "10"), ("10", "1"), ("10", "8"), ("12", "10")]
        # self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
        #                   ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]
        # self.scenarios = [('35', '31'), ('7', '18'), ('20', '30'), ('6', '19'), ('18', '23')]  # 嘉文师姐
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        self.mid_channels = 64
        self.final_out_channels = 64
        self.features_len = 64

        # TCN features
        self.tcn_layers = [75, 150, 300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 100
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 100
        self.hidden_dim = 500
        self.prej_input_channels = 1


class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()

        self.data_base_path = './datasets/HHAR'
        self.window_size = 100
        self.input_dim = 3
        self.class_num = 6
        self.time_interval = 6
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)


        self.sequence_len = 100
        # self.scenarios = [("2", "7"), ("5", "0")]
        self.scenarios = [("1", "6"), ("2", "7"),  ("4", "5"),("3", "8"),("0", "6"),
                          ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")]
        # self.scenarios = [("1", "5"),("1", "6"), ("1", "7"), ("1", "8"),  ("2", "1")]
        # self.scenarios = [('3', '1'), ('7', '1'), ('5', '1'), ('4', '1'), ('8', '6'), ('1', '5'), ('6', '1'),
        #                   ('6', '8'), ('1', '7'), ('5', '8'), ('5', '6'), ('7', '5'), ('8', '5'), ('3', '7'),
        #                   ('2', '6'), ('6', '5'), ('2', '5'), ('8', '1'), ('1', '6'), ('7', '6'), ('2', '1'),
        #                   ('2', '8'), ('4', '7'), ('1', '8'), ('4', '5'), ('3', '5'), ('7', '3'), ('5', '7'),
        #                   ('4', '3'), ('8', '2'), ('5', '2'), ('7', '8'), ('3', '6'), ('6', '2')]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 100
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 100
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 100
        self.hidden_dim = 500


class FD(object):
    def __init__(self):
        super(FD, self).__init__()

        self.data_base_path = './datasets/FD'
        self.window_size = 6
        self.input_dim = 15
        self.class_num = 3
        self.time_interval = 16
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)


        self.sequence_len = 5120
        self.scenarios = [("0", "1"), ("0", "3"), ("1", "0"), ("1", "2"),("1", "3"),
                          ("2", "1"),("2", "3"),  ("3", "0"), ("3", "1"), ("3", "2")]
        # self.scenarios = [('1', '3'), ('3', '1'), ('2', '3'), ('0', '3'), ('3', '2'), ('0', '1'), ('1', '2'),
        #                   ('0', '2'), ('2', '1'), ('1', '0'), ('3', '0'), ('2', '0')]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 100
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 100
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 100
        self.hidden_dim = 500


class AIR(object):
    def __init__(self):
        super(AIR, self).__init__()
        self.class_names = []
        self.sequence_len = 100
        self.scenarios = [
            ("BeiJing", "GuangZhou"), ("BeiJing", "TianJin"), ("BeiJing", "ShenZhen"),
            ("GuangZhou", "BeiJing"), ("GuangZhou", "TianJin"), ("GuangZhou", "ShenZhen"),
            ("TianJin", "BeiJing"), ("TianJin", "GuangZhou"), ("TianJin", "ShenZhen"),
            ("ShenZhen", "BeiJing"), ("ShenZhen", "GuangZhou"), ("ShenZhen", "TianJin")
        ]
        self.num_classes = 1
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 11
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.1

        # features
        self.mid_channels = 64
        self.final_out_channels = 100
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150, 300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 100
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 100
        self.hidden_dim = 500

        # signature
        self.depth = 3


class PPG(object):
    def __init__(self):
        super(PPG, self).__init__()
        self.class_names = []
        self.sequence_len = 100
        # cycling drive sitting walking
        self.scenarios = [
            ("cycling", "drive"), ("cycling", "sitting"), ("cycling", "walking"),
            ("drive", "cycling"), ("drive", "sitting"), ("drive", "walking"),
            ("sitting", "cycling"), ("sitting", "drive"), ("sitting", "walking"),
            ("walking", "cycling"), ("walking", "drive"), ("walking", "sitting")
        ]
        self.num_classes = 11
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 11
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.1

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150, 300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        # signature
        self.depth = 3
