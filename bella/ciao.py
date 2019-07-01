#!/usr/bin/env python

import re
import gym
import numpy as np

from pprint import pprint

from api import ApiWrapper
from config import MAX_ALERTS_PER_HOST


class GemelState(object):
    pass


class GemelEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.simulations = None
        self.ip_id_map = None
        self.known_alerts = None
        self.vnets = None

    @property
    def _interval(self):
        return 600

    def _fetch_alerts(self):

        ids_alerts = ApiWrapper.get_events(interval=self._interval)

        alerts = [(alert["src"], int(re.match(r".*:(\d+):.*", alert["sig_name"]).group(1)))
                  for net, alerts in ids_alerts.items() for alert in alerts]

        obs = {}
        for src_ip, alert_code in alerts:
            obs[src_ip] = obs.get(src_ip, []) + [alert_code]

        return dict([(k, v[-MAX_ALERTS_PER_HOST:]) for k, v in obs.items()])

    def _get_ids_observations(self):

        # get list of alerts per host
        ids_info = {self.ip_id_map[k]: v for k, v in self._fetch_alerts().items() if k in self.ip_id_map}

        # convert to sorted list
        ids_info = [ids_info[k] for k in sorted(ids_info.keys())]

        # convert to one-hot notation
        ids_info = [
            [
                [1 if alert == a["id"] else 0 for a in self.known_alerts]
                for alert in alert_list
            ]
            for alert_list in ids_info
        ]

        # convert to NumPy n-dimensional array
        return np.asarray(ids_info)

    def _get_vnet_status(self):
        vn_status = ApiWrapper.vnet_status()
        return vn_status

    def _get_state(self):
        return self._get_vnet_status(), self._get_ids_observations()

    def _init_net_info(self):
        """
        Fetches Gemel SDN network info from the API and
        initiates essential info such as number of vnets,
        number of simulations, etc
        """

        # fetch list of simulation hosts
        sims_list = ApiWrapper.get_sims()
        idx = 0
        ip_id_map = {}

        # assign a zero-based ID to each mac-address and store
        # the mapping
        for _, hosts in sims_list.items():
            for host in hosts:
                host["id"] = idx
                ip_id_map[host["overlay_ip"]] = idx
                idx += 1

        # fetch list of know alerts and sort by ID
        alerts = ApiWrapper.get_known_alert()
        alerts = sorted(alerts, key=lambda x: x["id"])

        # fetch vnet list
        self.vnets = ApiWrapper.vnet_list()

        self.simulations = sims_list
        self.host_count = idx
        self.ip_id_map = ip_id_map
        self.known_alerts = alerts

    def _reset_all_hosts_vnet(self):
        """
        Moves all hosts to the initial virtual-net (lowest security)
        """
        for _, hosts in self.simulations.items():
            for host in hosts:
                ApiWrapper.set_vnet(host["mac"], self.vnets[0]["name"])

    def step(self, action):
        pass

    def reset(self):
        self._init_net_info()
        self._reset_all_hosts_vnet()
        return self._get_state()

    def render(self, mode='human', close=False):
        pass


if __name__ == "__main__":
    env = GemelEnv()
    env.reset()

