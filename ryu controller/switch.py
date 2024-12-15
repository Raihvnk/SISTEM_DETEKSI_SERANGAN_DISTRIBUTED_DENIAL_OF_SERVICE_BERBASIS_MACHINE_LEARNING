from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.lib.packet import ipv4, tcp, udp, icmp
import joblib
import pandas as pd
import time
import socket
import struct
from collections import defaultdict

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.model = joblib.load('model.pkl')
        self.feature_names = joblib.load('model_features.pkl')
        self.packet_stats = defaultdict(lambda: {'packet_count': 0, 'byte_count': 0, 'start_time': time.time()})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.request_stats(datapath)

    def request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        for stat in body:
            match = stat.match
            ipv4_src = match.get('ipv4_src')
            ipv4_dst = match.get('ipv4_dst')
            ip_proto = match.get('ip_proto')

            if ipv4_src and ipv4_dst:
                flow_id = hash((ipv4_src, ipv4_dst, match.get('tcp_src', 0), match.get('tcp_dst', 0), ip_proto))
                flow_duration_sec = stat.duration_sec + stat.duration_nsec / 1e9
                packet_count = stat.packet_count
                byte_count = stat.byte_count

                packet_count_per_second = packet_count / flow_duration_sec if flow_duration_sec > 0 else 0
                byte_count_per_second = byte_count / flow_duration_sec if flow_duration_sec > 0 else 0

                self.packet_stats[(ipv4_src, ipv4_dst)] = {
                    'packet_count': packet_count,
                    'byte_count': byte_count,
                    'start_time': time.time() - flow_duration_sec
                }

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle=300, hard=600):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst,
                                idle_timeout=idle, hard_timeout=hard, buffer_id=buffer_id or ofproto.OFP_NO_BUFFER)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPErrorMsg, MAIN_DISPATCHER)
    def error_msg_handler(self, ev):
        msg = ev.msg
        self.logger.error('OFPErrorMsg received: type=0x%02x code=0x%02x '
                          'message=%s',
                          msg.type, msg.code, msg.data)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            ip_proto = ip_pkt.proto

            tp_src = 0
            tp_dst = 0
            flags = 0
            icmp_code = 0
            icmp_type = 0

            if tcp_pkt := pkt.get_protocol(tcp.tcp):
                tp_src = tcp_pkt.src_port
                tp_dst = tcp_pkt.dst_port
                flags = tcp_pkt.bits
            elif udp_pkt := pkt.get_protocol(udp.udp):
                tp_src = udp_pkt.src_port
                tp_dst = udp_pkt.dst_port
            elif icmp_pkt := pkt.get_protocol(icmp.icmp):
                icmp_code = icmp_pkt.code
                icmp_type = icmp_pkt.type

            total_length = ip_pkt.total_length
            flow_id = hash((src_ip, dst_ip, tp_src, tp_dst, ip_proto))

            if (src_ip, dst_ip) not in self.packet_stats:
                self.packet_stats[(src_ip, dst_ip)]['start_time'] = time.time()

            flow_duration_sec = time.time() - self.packet_stats[(src_ip, dst_ip)]['start_time']
            packet_count = self.packet_stats[(src_ip, dst_ip)]['packet_count'] + 1
            byte_count = self.packet_stats[(src_ip, dst_ip)]['byte_count'] + total_length

            self.packet_stats[(src_ip, dst_ip)]['packet_count'] = packet_count
            self.packet_stats[(src_ip, dst_ip)]['byte_count'] = byte_count

            packet_count_per_second = packet_count / flow_duration_sec if flow_duration_sec > 0 else 0
            byte_count_per_second = byte_count / flow_duration_sec if flow_duration_sec > 0 else 0

            def ip_to_int(ip):
                return struct.unpack("!I", socket.inet_aton(ip))[0]

            src_ip_int = ip_to_int(src_ip)
            dst_ip_int = ip_to_int(dst_ip)

            input_data = [time.time(), dpid, flow_id, src_ip_int, tp_src, dst_ip_int, tp_dst, ip_proto, icmp_code, icmp_type,
                          flow_duration_sec, 0, 0, 0, flags, packet_count, byte_count,
                          packet_count_per_second, 0, byte_count_per_second, 0]
            input_df = pd.DataFrame([input_data], columns=self.feature_names)

            # Prediction and Logging
            prediction = self.model.predict(input_df)[0]
            traffic_type = 'DDoS' if prediction == 1 else 'Normal'
            self.logger.info(f"Traffic Type: {traffic_type}, Source IP: {src_ip}, Destination IP: {dst_ip}")

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

