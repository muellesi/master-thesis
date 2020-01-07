from tkinter import *
from tkinter.ttk import *

from ..actions.action_manager import ActionManager
from ..gesture_item import GestureItem



class MainWindow():

    def __init__(self, action_manager: ActionManager, sample_record_callback, main_app_callback, save_gestures_callback):
        self.alive = True

        # Main Window
        self._root_wnd = Tk()
        self._root_wnd.title = "Gesture Control"
        self._root_wnd.resizable(0, 0)
        self._root_wnd.wm_attributes("-topmost", 1)
        self._root_wnd.protocol("WM_DELETE_WINDOW", self._on_quit)

        ###### Internal variables #####
        self._action_manager = action_manager
        self._gestures = []

        self._last_selected_gesture_index = None
        self._last_selected_sample_index = None
        self._name_variable = StringVar(self._root_wnd)
        self._action_variable = StringVar(self._root_wnd)
        self._sample_record_callback = sample_record_callback
        self._main_app_callback = main_app_callback
        self._save_gestures_callback = save_gestures_callback

        # Gesture List
        self._gesture_list_frame = Frame(self._root_wnd)
        self._gesture_list_frame.grid(row = 0, column = 0)

        Label(self._gesture_list_frame, text = "Gesture classes").grid(row = 0, columnspan = 2)

        self._gesture_list = Listbox(self._gesture_list_frame)
        self._gesture_list.grid(row = 1, columnspan = 2)

        self._remove_gesture_button = Button(self._gesture_list_frame, text = "Remove")
        self._remove_gesture_button.grid(row = 2, column = 1)

        # Details
        details_frame = Frame(self._root_wnd)
        details_frame.grid(row = 0, column = 1)

        # Details / Settings
        Label(details_frame, text = "Settings").grid(row = 0, column = 0)
        settings_frame = Frame(details_frame)
        settings_frame.grid(row = 1, column = 0)

        name_frame = Frame(settings_frame)
        name_frame.grid(row = 1, column = 0)
        Label(name_frame, text = "  Name: ").grid(row = 0, column = 0)
        tb_name = Entry(name_frame, textvariable = self._name_variable)
        tb_name.grid(row = 0, column = 1)

        action_frame = Frame(settings_frame)
        action_frame.grid(row = 2, column = 0)
        Label(action_frame, text = "Action: ").grid(row = 0, column = 0)
        sb_action = OptionMenu(action_frame, self._action_variable, *self._action_manager.get_action_names())
        sb_action.grid(row = 0, column = 1)

        self._save_gesture_button = Button(settings_frame, text = "Save",
                                           command = lambda: self._save_gesture(self._name_variable.get(),
                                                                                self._action_variable.get()))
        self._save_gesture_button.grid(row = 5, column = 0)

        # Details / Gesture samples
        self._gesture_sample_list_frame = Frame(details_frame)
        self._gesture_sample_list_frame.grid(row = 1, column = 1)

        Label(details_frame,
              text = "Class samples").grid(row = 0, column = 1)

        # exportselection = False is needed to prevent selection from gesture box to be deactivated,
        # see https://stackoverflow.com/questions/10048609/how-to-keep-selections-highlighted-in-a-tkinter-listbox
        self._gesture_sample_list = Listbox(self._gesture_sample_list_frame, exportselection = False)
        self._gesture_sample_list.grid(row = 1, columnspan = 2)

        self._add_gesture_sample_button = Button(self._gesture_sample_list_frame, text = "Add...",
                                                 command = lambda: self._add_gesture_sample(
                                                        self._gesture_list.curselection()[0]))
        self._add_gesture_sample_button.grid(row = 2, column = 0)

        self._remove_gesture_sample_button = Button(self._gesture_sample_list_frame, text = "Remove",
                                                    command = lambda: self._remove_gesture_sample())
        self._remove_gesture_sample_button.grid(row = 2, column = 1)

        self._start_button = Button(self._root_wnd, text = "Start Application", command = lambda : self._main_app_callback(self._gestures))
        self._start_button.grid(row = 1, columnspan = 2)

        # INIT application state
        self._last_selected_gesture_index = self._gesture_list.curselection()
        self._last_selected_sample_index = self._gesture_sample_list.curselection()



    def _on_quit(self):
        self.alive = False
        self._root_wnd.destroy()


    def update(self):
        self._root_wnd.update_idletasks()
        self._root_wnd.update()

        if self.alive:
            # If there are no defined gestures, we have to do nothing
            if self._gesture_list.size() > 0:
                selected_gesture = self._gesture_list.curselection()
                if self._last_selected_gesture_index != selected_gesture:
                    self._onGestureSelectChanged()
                    self._last_selected_gesture_index = selected_gesture

            if self._gesture_sample_list.size() > 0:
                selected_sample = self._gesture_sample_list.curselection()
                if self._last_selected_sample_index != selected_sample:
                    self._onSampleSelectChanged()
                    self._last_selected_sample_index = self._gesture_sample_list.curselection()
            else:
                self._remove_gesture_sample_button.config(state = 'disabled')


    def set_gestures(self, gestures):
        self._gestures = gestures
        self._update_gesture_list()


    def _update_gesture_list(self):
        current_index = 0
        if self._gesture_list.size() > 0:
            current_index = 0 if not len(self._gesture_list.curselection()) > 0 else self._gesture_list.curselection()
            self._gesture_list.delete(0, END)

        for gesture in self._gestures:
            self._gesture_list.insert(END, gesture.name)

        if self._gesture_list.size() > 0:
            self._gesture_list.select_set(current_index)
            selection = self._gesture_list.curselection()
            if len(selection) > 0:
                self._update_sample_list(selection[0])


    def _update_sample_list(self, gesture_index):
        if self._gesture_sample_list.size() > 0:
            self._gesture_sample_list.delete(0, END)

        if gesture_index < len(self._gestures):
            i = 1
            for sample in self._gestures[gesture_index].samples:
                self._gesture_sample_list.insert(END, str(i))
                i += 1


    def _update_action_list(self):
        self._action_manager.get_action_names()


    def _save_gesture(self, name, action):
        for gesture in self._gestures:
            if gesture.name == name:
                gesture.action = action
                self._update_gesture_list()
                self._save_gestures_callback(self._gestures)
                return
        self._gestures.append(GestureItem(name, [], action))
        self._update_gesture_list()
        self._save_gestures_callback(self._gestures)

    def _onGestureSelectChanged(self):
        if not self._gesture_list.size() > 0:
            return
        selection = self._gesture_list.curselection()
        if len(selection) > 0:
            selection = selection[0]
            self._remove_gesture_button.config(state = 'enabled')
            self._name_variable.set(self._gestures[selection].name)
            self._action_variable.set(self._gestures[selection].action)
            self._update_sample_list(selection)
            self._add_gesture_sample_button.config(state = 'enabled')
        else:
            self._name_variable.set("")
            self._action_variable.set(None)
            self._remove_gesture_button.config(state = 'disabled')
            self._add_gesture_sample_button.config(state = 'disabled')


    def _onSampleSelectChanged(self):
        if not self._gesture_sample_list.size() > 0:
            return
        selection = self._gesture_sample_list.curselection()
        if len(selection) > 0:
            selection = selection[0]
            self._remove_gesture_sample_button.config(state = 'enabled')
        else:
            self._remove_gesture_sample_button.config(state = 'disabled')


    def _add_gesture_sample(self, gesture):
        samples = self._sample_record_callback()
        selected_gesture = self._gesture_list.curselection()[0]
        self._gestures[selected_gesture].samples.append(samples)
        self._update_sample_list(selected_gesture)
        self._save_gestures_callback(self._gestures)


    def _remove_gesture_sample(self):
        selected_gesture = self._gesture_list.curselection()[0]
        selected_sample = self._gesture_sample_list.curselection()[0]
        print(self._gestures[selected_gesture].samples)
        self._gestures[selected_gesture].samples.pop(selected_sample)
        print(self._gestures[selected_gesture].samples)
        self._update_sample_list(selected_gesture)
        self._save_gestures_callback(self._gestures)
