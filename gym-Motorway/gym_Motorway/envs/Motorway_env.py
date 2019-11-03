
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
from gym.envs.classic_control import rendering

import math
import numpy as np
import baseconvert

import pyglet
from pyglet import gl
	
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

# Created by Stefano Sabatini. 
# stefano.sabatini90@gmai.com


FPS = 50

STATE_W = 90   # less than Atari 160x192
STATE_H = 180
VIDEO_W = 300
VIDEO_H = 600
WINDOW_W = 500
WINDOW_H = 1000

# Environemnt param
LANE_NUMBER  = 4
LANE_WIDTH   = 5
ZOOM         = 10        # Camera zoom
TRACK_LENGTH = 2000
PLAYFIELD_W  = LANE_NUMBER*LANE_WIDTH + 500

# Car Param
CAR_WIDTH   = 2;
CAR_LENGTH  = 5;
DRAG_ACC    = 0.2;
GRASS_VISC_FRICTION = 0.07;

MAX_SPEED_TRAFFIC  = 230 
MIN_SPEED_TRAFFIC  = 90

MAX_SPEED  = 230 
MIN_SPEED  = 60

# TRAFFIC PARAM
TRAFFIC_PARTICIPANTS_DENSITY = 35 # num participants / km
SAFE_DIST = 10

class Car:    
    id = 0
    def __init__(self,start_s,start_lane_idx, start_v = 0,color = (1,0,0)):
        self.s          =  start_s
        self.s_old      =  start_s
        self.lane_idx   =  start_lane_idx
        self.v          =  start_v
        self.color      =  color
        self.rnd_acc    =  0
        self.car_id     =  Car.id
        Car.id = Car.id+1
        
    def step(self,dt,action = [0,0], coasting = False, traffic = []):  
        if coasting:
            # add some random acceleration
            sample = np.random.uniform(0,1,1)
            v_span = (self.v - MIN_SPEED_TRAFFIC)/(MIN_SPEED_TRAFFIC-MAX_SPEED_TRAFFIC)
            acc_probability   = 1-0.05*(1-v_span)
            break_probability = 0.05*v_span
            
            if sample > acc_probability:
                self.rnd_acc = np.random.uniform(-10,-3,1)
            elif sample < break_probability:
                self.rnd_acc = np.random.uniform(3,10,1)  
#            elif sample > :
#                self.rnd_acc = 0
            else:    
                pass
                           
#            
            self.v = min(max(self.v + self.rnd_acc*dt,MIN_SPEED_TRAFFIC/3.6),MAX_SPEED_TRAFFIC/3.6)
            
            for tp in traffic:
                if (tp.s - self.s) < SAFE_DIST and (tp.s - self.s) > 0 and (tp.lane_idx == self.lane_idx):
                    self.v = min(tp.v,self.v) # ACC
                    break
            self.s    = self.s + dt*self.v 
            
        else:    
            # lateral lane change
            self.lane_idx = min(max(self.lane_idx + action[0],-1),LANE_NUMBER)
            # longitudinal movment
            if self.lane_idx < 0 or self.lane_idx >= LANE_NUMBER:
                self.v    = min(max(self.v + action[1] - GRASS_VISC_FRICTION*self.v ,0), MAX_SPEED)
            else:
                self.v    = min(max(self.v + action[1] - DRAG_ACC ,0), MAX_SPEED/3.6)            
            self.s    = self.s + dt*self.v
        
    def render(self,viewer,pos):
        self.car_render = rendering.FilledPolygon( [
                    (-CAR_WIDTH/2,+CAR_LENGTH/2), (+CAR_WIDTH/2,+CAR_LENGTH/2),
                    (+CAR_WIDTH/2,-CAR_LENGTH/2), (-CAR_WIDTH/2,-CAR_LENGTH/2)
                    ])
        self.car_render.set_color(self.color[0],self.color[1],self.color[2])
        self.car_transf = rendering.Transform()
        self.car_render.add_attr(self.car_transf)
        viewer.add_geom(self.car_render)
        self.car_transf.set_translation(pos[0], pos[1])
        

        
        

class MotorwayEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }
    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.traffic_participants = []        
        self.action = []
        
        self.possible_long_commands = {0:'COASTING',1:'ACCELERATE',2:'BREAKING'}
        self.possible_lat_commands  = {0:'KEEP_LANE',1:'LANE_CHANGE_LEFT',2:'LANE_CHANGE_RIGTH'}
        self.command2acc_mapping    = {'COASTING':0,'ACCELERATE':1,'BREAKING':-2}
        self.command2steer_mapping  = {'KEEP_LANE':0,'LANE_CHANGE_LEFT':-1,'LANE_CHANGE_RIGTH':1}
        self.action_space           = spaces.Discrete(len(self.possible_lat_commands)*len(self.possible_long_commands)) 
        self.observation_space      = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        
    def reset(self):
        Car.id = 0 # reset traffic id count
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0
        self.car = Car(start_s = 0, start_lane_idx = math.floor(LANE_NUMBER/2))
        self.traffic_participants = []
        self.spawn_traffic()

        return self.step(None)[0]

    def step(self, action):    
        

            
        dt = 1/FPS
        step_reward = 0
        done = False

        if action is not None:
            
            # map action number to car command (acceleration/deceleration and steering)
            command = self.action2command(action)       
            lat_command  = self.possible_lat_commands[command[0]]
            long_command = self.possible_long_commands[command[1]]
            car_commands = [self.command2steer_mapping[lat_command],self.command2acc_mapping[long_command]]
            
            self.car.step(dt,car_commands,coasting = False)
        
        # update traffic
        if self.t>1:
            for tp in self.traffic_participants:
                tp.step(dt,coasting = True, traffic = self.traffic_participants)
                
                if self.check_collision(self.car,tp):
                    print("COLLISION!!! ")
                    self.reward -= 1000
                    done = True
                
        self.t += 1.0/FPS             
        self.state = self.render("state_pixels")

        if action is not None: # First step without action, called from reset()
            self.reward +=-1 
            if (self.car.s - self.car.s_old>1):
                self.car.s_old = self.car.s
                self.reward +=100
            if self.car.lane_idx < 0 or self.car.lane_idx >= LANE_NUMBER:
                self.reward -= 100
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.car.s > TRACK_LENGTH:
                print("FINISHED TRACK!!!!")
                done = True
                
        if self.t > 60:
            done = True
            
        return self.state, step_reward, done, {}
    
    def action2command(self,action):
        command = np.asarray(baseconvert.base(action,10,3))
        if len(command) == 1:
            command = np.append(0,command)
        return command
        
    def command2action(self,command):  
        return np.asarray(baseconvert.base(command,3,10))
            
    def check_collision(self,car1,car2):
        return (car1.lane_idx == car2.lane_idx and abs(car1.s-car2.s) < CAR_LENGTH )
            
    def spawn_traffic(self):
        
        num_tp = TRAFFIC_PARTICIPANTS_DENSITY*(TRACK_LENGTH/1000)
        v     = np.random.uniform(70/3.6,120/3.6)
        lane  = round(np.random.uniform(0,LANE_NUMBER-1))
        s     = np.random.uniform(10,TRACK_LENGTH)
        color = np.random.uniform(0,1,3)
        color[0] = 1
        curr_tp = Car(s,lane,v,color)
        self.traffic_participants.append(curr_tp)
        
        tp_indx = 0
        while tp_indx < int(num_tp-1):
            v     = np.random.uniform(MIN_SPEED/3.6,MAX_SPEED/3.6)
            lane  = round(np.random.uniform(0,LANE_NUMBER-1))
            s     = np.random.uniform(15,TRACK_LENGTH/2)
            color = np.random.uniform(0,1,3)
            color[0] = 1
            curr_tp = Car(s,lane,v,color)
            
            skip_tp = False  
            for tp in self.traffic_participants:
                if self.check_collision(tp,curr_tp):
                    skip_tp = True
                    break
                
            if skip_tp:
                pass
            else:
                self.traffic_participants.append(curr_tp)
                tp_indx = tp_indx+1
                
                
                

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        
        # creating the view window
        if self.viewer is None:
            # Set up Viewer
#            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        # Prepare the roto translation to make the view point at the car s
        if mode == 'human':
            zoom = 0.1*max(1-self.t, 0) + ZOOM*min(self.t, 1)   # Animate zoom first second
        else:
            zoom = ZOOM
        car_x , car_y = self.curv2xy(self.car.s,self.car.lane_idx)
        angle = 0
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - ( 0 ),
            WINDOW_H/4 - ( + car_y*zoom ))
        self.transform.set_rotation(angle)
        
        # Draw Traffic Participant
        self.car.render(self.viewer,(car_x,car_y))
        
        for tp in self.traffic_participants:
            tp.render(self.viewer,self.curv2xy(tp.s,tp.lane_idx))
        
        
        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        if mode=='rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)
        gl.glViewport(0, 0, VP_W, VP_H)
        
        # FINALLY RENDER!
        self.transform.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        for geom in self.viewer.geoms:
            geom.render()
        self.transform.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)
        # clear geometries
        self.viewer.onetime_geoms = []
        self.viewer.geoms         = []
        
        # update the visualization if human playing
        if mode == 'human':
            win.flip()
            return self.viewer.isopen
        
        # getting a screenshot of the application to get the state array otherwise
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]
#        plt.imshow(arr)
        return arr
    
    
    def render_road(self): 
        
        
#         update the background only once every N seconds
        if (self.t % 2) <= 2/FPS :
            self.road_render_center = self.car.s 
#            print(" Re - rendering the road")
#            print(" .............")
#            print(" T = ", self.t)
            
        s = self.road_render_center
        tail = 200
        
        # render grass
        gl.glBegin(gl.GL_POLYGON)
        gl.glColor4f(0, 1, 0, 1.0)
        gl.glVertex3f(-PLAYFIELD_W/2, s -tail, 0)
        gl.glVertex3f(-PLAYFIELD_W/2, s +tail, 0)
        gl.glVertex3f(+PLAYFIELD_W/2, s +tail, 0)
        gl.glVertex3f(+PLAYFIELD_W/2, s -tail, 0)
        gl.glEnd()
        # render asphalt
        gl.glBegin(gl.GL_POLYGON)
        gl.glColor4f(0.66, 0.66, 0.66, 1.0)
        gl.glVertex3f(-LANE_NUMBER*LANE_WIDTH/2, s -tail, 0)
        gl.glVertex3f(-LANE_NUMBER*LANE_WIDTH/2, s +tail, 0)
        gl.glVertex3f(+LANE_NUMBER*LANE_WIDTH/2, s +tail, 0)
        gl.glVertex3f(+LANE_NUMBER*LANE_WIDTH/2, s -tail, 0)
        gl.glEnd()        
        # render lanes
        LINE_W = 0.3
        LINE_H = 1
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(1, 1, 1, 1.0)
        for l_idx in range(LANE_NUMBER+1):
            for s_idx in range(int(s-tail),int(s+tail),4):
                l_x, l_y = self.curv2xy(s_idx,l_idx)
                stripe_x = l_x - LANE_WIDTH/2 - LINE_W/2
                stripe_y = l_y
                gl.glVertex3f(stripe_x-LINE_W/2, stripe_y-LINE_H/2, 0)
                gl.glVertex3f(stripe_x-LINE_W/2, stripe_y+LINE_H/2, 0)
                gl.glVertex3f(stripe_x+LINE_W/2, stripe_y+LINE_H/2, 0)
                gl.glVertex3f(stripe_x+LINE_W/2, stripe_y-LINE_H/2, 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        true_speed = self.car.v
        vertical_ind(15, 0.02*true_speed, (1,1,1))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()    

    def curv2xy(self,s,lane_idx):
        if (LANE_NUMBER % 2) == 0: 
            x = LANE_WIDTH*(-LANE_NUMBER/2+0.5+lane_idx)
            y = s
        else:                
            x = LANE_WIDTH*(-math.floor(LANE_NUMBER/2)+lane_idx)
            y = s
        return x , y
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0 ] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = 1
        if k==key.RIGHT: a[0] = 2
        
        if k==key.UP:    a[1] = 1
        if k==key.DOWN:  a[1] = 2   
        
    def key_release(k, mod):
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[1] = 0
        
    env = MotorwayEnv()
    env.render()
    env.viewer.window.on_key_press   = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    reset_steering = False
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            a_prev = a
            if a[0] != 0:
                reset_steering = True
            s, r, done, info = env.step(env.command2action(a))
            if reset_steering: # reset steering to avoid multiple lane change for long button press
                a[0] = 0
                reset_steering = False
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
#                import matplotlib.pyplot as plt
#                plt.imshow(s)
#                plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
