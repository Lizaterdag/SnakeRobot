import natnet
client = natnet.Client.connect("192.168.1.132")
print("lalala")
#client = natnet.Client.connect()
client.set_callback(
    lambda rigid_bodies, markers, timing: print(rigid_bodies))
print('set_callback done')
client.run_once()
print('done running')