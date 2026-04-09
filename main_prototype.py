import time
import keyboard

from state import State
import detection as det
import actions as act
import config as cfg
import cv2 as cv2


state = State.IDLE
running = False

last_menu_press = 0
last_action_time = 0


while True:

    # ---------- TOGGLE ----------
    if keyboard.is_pressed('t'):
        running = not running
        state = State.MENU if running else State.IDLE
        print("Running:", running)
        time.sleep(0.5)

    if not running:
        time.sleep(0.1)
        continue

    now = time.time()

    # ---------- GLOBAL OVERRIDES ----------

    if det.is_fight_active() and state != State.IN_FIGHT:
        print(">>> Fight detected → IN_FIGHT")
        state = State.IN_FIGHT
        last_action_time = 0

    elif state == State.MENU and det.is_vs_screen():
        print(">>> VS detected → VS_WAIT")
        state = State.VS_WAIT

    # ---------- FSM ----------

    if state == State.MENU:

        if det.is_menu() and (now - last_menu_press > cfg.MENU_PRESS_DELAY):
            print("Menu → Enter")
            act.press_menu_confirm()
            last_menu_press = now


    elif state == State.VS_WAIT:
        pass


    elif state == State.IN_FIGHT:

        if det.is_fight_over():
            print("Fight ended → MENU")
            state = State.MENU
            time.sleep(0.3)
            continue

        if not det.is_fight_active():
            print("Fight ended → MENU (fallback)")
            state = State.MENU
            time.sleep(0.3)
            continue

        if now - last_action_time > cfg.COMBAT_ACTION_DELAY:
            act.fight_combo()
            last_action_time = now


    elif state == State.IDLE:
        time.sleep(0.2)

    time.sleep(0.05)