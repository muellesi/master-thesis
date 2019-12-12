from tkinter import *
from tkinter.ttk import *
from ..actions.action_manager import ActionManager

class MainWindow():
    def __init__(self, action_manager:ActionManager):
        self.alive = True

        self.root_wnd = Tk()
        self.root_wnd.title = "Gesture Control"
        self.root_wnd.resizable(0,0)
        self.root_wnd.wm_attributes("-topmost", 1)
        self.root_wnd.protocol("WM_DELETE_WINDOW", self.on_quit)
        
        self.gesture_list_frame = Frame(self.root_wnd)
        self.gesture_list_frame.grid(row = 0, column = 0)
        
        Label(self.gesture_list_frame, text = "Gesture classes").grid(row = 0, columnspan = 2)
        
        self.gesture_list = Listbox(self.gesture_list_frame)
        self.gesture_list.grid(row = 1, columnspan = 2)
        
        self.add_gesture_button = Button(self.gesture_list_frame, text="Add gesture...")
        self.add_gesture_button.grid(row = 2, column = 0)

        self.remove_gesture_button = Button(self.gesture_list_frame, text = "Remove gesture")
        self.remove_gesture_button.grid(row = 2, column = 1)

        self.gesture_sample_list_frame = Frame(self.root_wnd)
        self.gesture_sample_list_frame.grid(row = 0, column = 1)

        Label(self.gesture_sample_list_frame, text = "Gesture class samples").grid(row = 0, columnspan = 2)

        self.gesture_sample_list = Listbox(self.gesture_sample_list_frame)
        self.gesture_sample_list.grid(row = 1, columnspan = 2)
        
        self.add_gesture_sample_button = Button(self.gesture_sample_list_frame, text="Add gesture sample...", command = lambda: self.action_creation_dialog)
        self.add_gesture_sample_button.grid(row = 2, column= 0)

        self.remove_gesture_sample_button = Button(self.gesture_sample_list_frame, text="Remove sample")
        self.remove_gesture_sample_button.grid(row = 2, column = 1)

        self.start_button = Button(self.root_wnd, text = "Start Application")
        self.start_button.grid(row = 1, columnspan=2)


        self._action_manager = action_manager

        ###### Internal variables #####
        self._gestures = None

        self._last_selected_gesture_index = None
        self._last_selected_sample_index  = None
    

    def on_quit(self):
        self.alive = False
        self.root_wnd.destroy()


    def update(self):
        self.root_wnd.update_idletasks()
        self.root_wnd.update()

        if self._last_selected_gesture_index != self.gesture_list.curselection():
            self._update_sample_list()

        self._last_selected_gesture_index = self.gesture_list.curselection()
        self._last_selected_sample_index  = self.gesture_sample_list.curselection()


    def set_gestures(self, gestures):
        self._gestures = gestures
        self._update_gesture_list()
        self._update_sample_list()


    def _update_gesture_list(self):
        self.gesture_list.delete(0, END)
        for gesture in self._gestures:
            self.gesture_list.insert(gesture.name, END)


    def _update_sample_list(self):
        selected_index = self.gesture_list.curselection()
        if selected_index < len(self._gestures):
            self.gesture_sample_list.delete(0, END)
            i = 1
            for sample in self._gestures[selected_index].samples:
                self.gesture_sample_list.insert(str(i), END)

    def update_action_list(self):
        self._action_manager.get_action_names()


    def action_creation_dialog(self, name = None, action = None):

        wnd = Toplevel()

        Label(wnd, text = "Name:").grid(row = 0, column = 0)
        name_variable = StringVar(wnd)
        tb_name = Entry(wnd, textvariable=name_variable)
        if name is not None:
            tb_name.insert(0, name)
        tb_name.grid(row = 0, column = 1)

        Label(wnd, text = "Action:").grid(row = 0, column = 0)
        action_variable = StringVar(wnd)
        if action and action in self._action_manager.get_action_names():
            action_variable.set(action)
        sb_action = OptionMenu(wnd, action_variable, *self._action_manager.get_action_names())
        sb_action.grid(row = 1, column = 1)

        return name_variable.get(), action_variable.get()
