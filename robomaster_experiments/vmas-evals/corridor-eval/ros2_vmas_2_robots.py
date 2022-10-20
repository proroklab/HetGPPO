
import math
# import torch
import os

import rclpy
from freyja_msgs.msg import CurrentState
from freyja_msgs.msg import ReferenceState
from rclpy.node import Node

import evaluate_model as MDEval


class MinimalPublisher(Node):
    #self.pn0 = 3.0;
    #self.pn1 = 1.0;
    #self.RS = ReferenceState();

    def __init__(self, style, a_range, v_range, model_name):
        super().__init__("lawnmower_publisher")

        self.a_range = a_range
        self.v_range = v_range
        self.model_name = model_name

        
        import torch

        dname = os.path.dirname(os.path.realpath(__file__))
        mdlpath = dname + f"/{self.model_name}"

        self.model = torch.load(mdlpath)
        self.model.eval()
        print("Model ready!")


        self.CS1 = ReferenceState()
        self.CS2 = ReferenceState()
        self.RS1 = ReferenceState()
        self.RS2 = ReferenceState()
        self.lastRS1 = ReferenceState()
        self.lastRS2 = ReferenceState()
       

        self.dt = 1.0 / 20.0
        self.mytime = 0
        self.subscriber1 = self.create_subscription(CurrentState, "/robomaster_1/current_state", self.cscb1, 1)
        self.publisher1 = self.create_publisher(ReferenceState, "/robomaster_1/reference_state", 1)
        self.subscriber2 = self.create_subscription(CurrentState, "/robomaster_2/current_state", self.cscb2, 1)
        self.publisher2 = self.create_publisher(ReferenceState, "/robomaster_2/reference_state", 1)
        timer_period = self.dt  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback);
        print( "ROS2 starting .." )



    def timer_callback(self):
        # evaluate model
        x = 1   # index into "pos" and "vel"
        y = 0
    
        pos1 = [self.CS1.pn, self.CS1.pe]
        vel1 = [self.CS1.vn, self.CS1.ve]
        pos2 = [self.CS2.pn, self.CS2.pe]
        vel2 = [self.CS2.vn, self.CS2.ve]

        print(pos1)
        print(pos2)
        
        cmd_vels = MDEval.compute_action_corridor( 
            pos1[x], 
            pos1[y], 
            vel1[x], 
            vel1[y], 
            pos2[x], 
            pos2[y], 
            vel2[x], 
            vel2[y], 
            model=self.model,
            u_range=self.v_range,
            deterministic=True,
        )

        cmd_vel1 = cmd_vels[0]
        cmd_vel2 = cmd_vels[1]
        # print( cmd_vels, cmd_vel1, cmd_vel2 );

        self.RS1.vn = cmd_vel1[x]
        self.RS1.ve = cmd_vel1[y]
        self.RS1.yaw = math.pi / 2 # math.pi / 4.0; # math.atan2( a[1], a[0] );
        self.RS1.an = self.RS1.ae = self.a_range
        self.RS1.header.stamp = self.get_clock().now().to_msg()

        self.RS2.vn = cmd_vel2[x]
        self.RS2.ve = cmd_vel2[y]
        self.RS2.yaw = 3 * math.pi / 2 # math.pi / 4.0; # math.atan2( a[1], a[0] );
        self.RS2.an = self.RS2.ae = self.a_range
        self.RS2.header.stamp = self.get_clock().now().to_msg()

        self.publisher1.publish( self.RS1 )
        self.publisher2.publish( self.RS2 )

        # print( [cmd_vel1[0]-self.RS1.ve, cmd_vel1[1]-self.RS1.vn], [cmd_vel2[0]-self.RS2.ve, cmd_vel2[1]-self.RS2.vn] );

        # book-keeping
        self.lastRS1 = self.RS1
        self.lastRS2 = self.RS2
        self.mytime += self.dt
    
    def cscb1( self, msg ):
      self.CS1.pn = msg.state_vector[0] #+ self.off_pn
      self.CS1.pe = msg.state_vector[1] # + self.off_pe
      self.CS1.vn = msg.state_vector[3]
      self.CS1.ve = msg.state_vector[4]
      #self.CS.yaw = msg.state_vector[8]
    def cscb2( self, msg ):
      self.CS2.pn = msg.state_vector[0] #+ self.off_pn
      self.CS2.pe = msg.state_vector[1] # + self.off_pe
      self.CS2.vn = msg.state_vector[3]
      self.CS2.ve = msg.state_vector[4]
      #self.CS.yaw = msg.state_vector[8]


def main(s):
    rclpy.init(args=None)

    minimal_publisher = MinimalPublisher(
        style=args.style,
        a_range=args.a_range,
        v_range=args.v_range,
        model_name=args.model_name,
    )

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Runs a vmas model.")

    parser.add_argument("--v_range", type=float)
    parser.add_argument("--a_range", type=float)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--style", type=int)

    args = parser.parse_args()

    main(args)