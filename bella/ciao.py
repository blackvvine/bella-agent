#!/usr/bin/env python
import enum
import re
import gym
import numpy as np

from gym.spaces import Discrete

from pprint import pprint

from bella.api import ApiWrapper
from bella.config import MAX_ALERTS_PER_HOST, MAX_STEPS_PER_EPISODE


class GemelState(object):
    pass


class GemelEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    class Reward(enum.Enum):
        PLACING = 1

    def __init__(self, reward=Reward.PLACING, max_steps=MAX_STEPS_PER_EPISODE, max_alerts=MAX_ALERTS_PER_HOST):

        self.simulations = None
        self.ip_id_map = None
        self.arp_table = None
        self.known_alerts = None
        self.vnets = None

        self.reward = reward
        self.current_step = 0
        self.max_steps = max_steps
        self.max_alerts_per_host = max_alerts

        self._init_net_info()

    @property
    def _interval(self):
        return 10

    def _fixate_feature_size(self, alert_list):
        cut = alert_list[-self.max_alerts_per_host:]
        padded = cut + (self.max_alerts_per_host - len(cut)) * [0]
        return padded

    def _fetch_alerts(self):
        """
        Fetch IDS alerts and filter out irrelevant one
        """

        ids_alerts = ApiWrapper.get_events(interval=self._interval)

        alerts = [(alert["src"], int(re.match(r".*:(\d+):.*", alert["sig_name"]).group(1)))
                  for net, alerts in ids_alerts.items() for alert in alerts]

        obs = {}
        for src_ip, alert_code in alerts:
            obs[src_ip] = obs.get(src_ip, []) + [alert_code]

        return obs

    def _get_ids_observations(self):
        """
        Get IDS alerts and convert to n-dimensional features array
        """

        # get list of alerts per host
        ids_info = {self.ip_id_map[k]: v for k, v in self._fetch_alerts().items() if k in self.ip_id_map}

        # add empty entries for absent hosts in the IDS alerts
        empty_lists = {k: [] for k in self.ip_id_map.values() if k not in ids_info.keys()}
        ids_info = {**ids_info, **empty_lists}

        # fix feature size (cut if more, pad if less)
        ids_info = {k: self._fixate_feature_size(v) for k, v in ids_info.items()}

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

    @property
    def _hosts_sorted_by_id(self):
        return sorted((host for _, hosts in self.simulations.items() for host in hosts), key=lambda x: x["id"])

    def _get_vnet_status(self):
        """
        Receives which vn each host is in and returns as
        feature array
        """

        # fetch where each host is
        vnet_status = ApiWrapper.vnet_status()

        # use list of vnet names to assign a "number" to each vnet
        # (i.e. index of the vnet in the list)
        vnet_names = [x["name"] for x in self.vnets]

        # get a list of vnet names for each host
        sorted_list = [vnet_status[host["mac"]] for host in self._hosts_sorted_by_id]

        # use vnet "number" instead of vnet name and convert to NumPy array
        return np.asarray([vnet_names.index(name) for name in sorted_list])

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
        for host_type, hosts in sims_list.items():
            for host in hosts:
                host["id"] = idx
                host["type"] = host_type
                ip_id_map[host["overlay_ip"]] = idx
                idx += 1

        # fetch list of know alerts and sort by ID
        alerts = ApiWrapper.get_known_alert()
        alerts = sorted(alerts, key=lambda x: x["id"])

        # fetch ARP table
        self.arp_table = ApiWrapper.get_arp_table()

        # fetch vnet list
        self.vnets = ApiWrapper.vnet_list()

        self.simulations = sims_list
        self.host_count = idx
        self.ip_id_map = ip_id_map
        self.known_alerts = alerts

        self.action_space = Discrete(len(self._hosts_sorted_by_id) + 1)

    def _get_mac_id(self, mac):
        for x in ((ip, mac) for ip, macs in self.arp_table.items() for mac in macs):
            if x[1] == mac:
                return self.ip_id_map[x[0]]
        raise Exception(f"MAC address {mac} not found in ARP table")

    def _reset_all_hosts_vnet(self):
        """
        Moves all hosts to the initial virtual-net (lowest security)
        """
        for host in self._hosts_sorted_by_id:
                ApiWrapper.set_vnet(host["mac"], self.vnets[0]["name"])

    def _apply_action(self, action):
        sims = self._hosts_sorted_by_id
        ApiWrapper.toggle(sims[action]["mac"])

    def _is_terminal(self):
        """
        Returns whether the "terminal" state has reached. In our current
        problem statement, this is a continuous space so "terminal" is
        just a bad name for max steps reached
        """
        return self.current_step > self.max_steps

    # noinspection PyRedundantParentheses
    def step(self, action):
        """
        :param action:
        :return: 3-tuple containing:
            0 - new state
            1 - reward from the action taken
            2 - whether the new state is terminal
        """

        if isinstance(action, np.integer):
            action = int(action)

        assert isinstance(action, int)

        self.current_step += 1

        # the non-NOP action
        if action < len(self._hosts_sorted_by_id):
            # apply the toggle action
            self._apply_action(action)

        return (self._get_state(), self._get_reward(), self._is_terminal())

    def _get_reward(self):
        if self.reward == GemelEnv.Reward.PLACING:
            sims = self._hosts_sorted_by_id
            reward_ = 0
            for idx, vnet_id in enumerate(self._get_vnet_status()):
                host = sims[idx]
                vnet = self.vnets[vnet_id]
                reward_ += (-1 if host["type"] == "benign" else +1) * vnet["security_level"]
            return reward_
        else:
            raise Exception(f"Unknown reward scheme {self.reward}")

    def observation_shape(self):

        flat_size = len(self._hosts_sorted_by_id) + \
               len(self._hosts_sorted_by_id) * len(self.known_alerts) * self.max_alerts_per_host

        # noinspection PyRedundantParentheses
        return (flat_size,)

    def state(self):
        return self._get_state()

    def reset(self):
        self._init_net_info()
        self._reset_all_hosts_vnet()
        self.current_step = 0
        return self._get_state()

    def render(self, mode='human', close=False):
        pass


if __name__ == "__main__":
    env = GemelEnv()
    pprint(env.state()[1])



