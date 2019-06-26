import requests

from bella.config import HALSEY_BASE_URL


class ApiWrapper(object):

    TOGGLE = HALSEY_BASE_URL + "/vnet/toggle"
    STATS = HALSEY_BASE_URL + "/sim/attack"
    QOS = HALSEY_BASE_URL + "/sim/qos"
    HIST = HALSEY_BASE_URL + "/ids/hist"
    VNET = HALSEY_BASE_URL + "/vnet/get?host="

    EVENTS = HALSEY_BASE_URL + "/ids/events"
    ARP = HALSEY_BASE_URL + "/topo/arp"
    SIMS = HALSEY_BASE_URL + "/topo/sims"

    @classmethod
    def get_events(cls, interval=60):
        """
        :param interval: get events for last X seconds
        """
        return requests.get(cls.EVENTS, {"interval": interval}).json()

    @classmethod
    def get_arp_table(cls):
        return requests.get(cls.ARP).json()

    @classmethod
    def get_sims(cls):
        return requests.get(cls.SIMS).json()


