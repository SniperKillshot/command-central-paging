"""DVM (Digital Voice Modem) FNE peer support.

Command Central Paging connects to a DVMProject FNE (dvmfne, from
https://github.com/DVMProject/dvmhost) as an ordinary network peer, the same
way a dvmhost repeater or dvmbridge does.
"""
from src.dvm.peer import FNEPeer, PeerState

__all__ = ['FNEPeer', 'PeerState']
