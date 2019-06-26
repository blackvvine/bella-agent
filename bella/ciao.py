#!/usr/bin/env python

import re
import gym

from api import ApiWrapper
from config import MAX_ALERTS_PER_HOST


class GemelState(object):
    pass


class GemelEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.simulations = None
        self.ip_id_map = None

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

    def _get_observations(self):

        ids_info = self._fetch_alerts()

        for ip, alerts in ids_info.items():
            if ip not in self.ip_id_map:
                continue

    def _init_net_info(self):

        sims_list = ApiWrapper.get_sims()
        idx = 0
        ip_id_map = {}

        for type, hosts in sims_list.items():
            for host in hosts:
                host["id"] = idx
                ip_id_map[host["id"]] = idx
                idx += 1

        self.simulations = sims_list
        self.host_count = idx
        self.ip_id_map = ip_id_map

    def step(self, action):
        pass

    def reset(self):
        self._init_net_info()

    def render(self, mode='human', close=False):
        pass


if __name__ == "__main__":
    env = GemelEnv()
    obs = env._fetch_observation()
    import pprint
    pprint.PrettyPrinter().pprint(obs)

