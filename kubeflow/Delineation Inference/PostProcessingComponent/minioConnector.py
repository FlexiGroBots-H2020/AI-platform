import os
from minio import Minio
import glob

class minioConector:
    def __init__(self):

        self.host = 'minio-cli.platform.flexigrobots-h2020.eu'
#         self.access = 'Pilot3-FlexiGroBotsH2020'
#         self.secret ='jqFRgK!81XOt2H!'
        self.access = 'flexigrobot2020_Admin'
        self.secret = '7*v67fhM9902^P5L5@Nl'


#         self.host = os.getenv('MINIO_HOST')
#         self.access = os.getenv('MINIO_ACCESS_KEY')
#         self.secret =os.getenv('MINIO_SECRET_KEY')
        print(self.host)
        print(self.access)
        print(self.secret)
        self.connection_with_minio = Minio(self.host,
            access_key= self.access,
            secret_key=self.secret,
            secure=True)


    #Files management


    def uploadFiles(self, bucket_name, my_object_name, my_filename_path):
        if bucket_name in self.listBuckets():
            self.connection_with_minio.fput_object(bucket_name, my_object_name, my_filename_path)
        else:
            self.createBuckets(bucket_name)
            self.connection_with_minio.fput_object(bucket_name, my_object_name, my_filename_path)

    def uploadDirectory(self, local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows

            if not os.path.isfile(local_file):
                self.uploadDirectory(local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(os.sep, "/")  # Replace \ with / on Windows
                self.connection_with_minio.fput_object(bucket_name, remote_path, local_file)


    #Buckets management
    def downloadBucket(self, bucket_name, file_path):
        try:
            for item in self.connection_with_minio.list_objects(bucket_name,  recursive=True):
                self.connection_with_minio.fget_object(bucket_name, item.object_name, os.path.join(file_path,item.object_name))
        except:
            print("Object storage not reachable")

    def createBuckets(self, Bucket_name):
        if self.existBuckets(Bucket_name) == True:
            print(Bucket_name,"bucket already exists")
        else:
            self.connection_with_minio.make_bucket(Bucket_name)

    def removeBuckets(self, Bucket_name):
        if self.existBuckets(Bucket_name) == True:

            objects = self.listObjects(Bucket_name)
            if objects:
                for obj in objects:
                    print(obj.object_name)
                    self.deleteOjects(Bucket_name, obj.object_name)
                self.connection_with_minio.remove_bucket(Bucket_name)
            else:
                self.connection_with_minio.remove_bucket(Bucket_name)
        else:
            print("Not exist:", Bucket_name) #Put assert instead print

    def listBuckets(self):
        return list(self.connection_with_minio.list_buckets())
    def existBuckets(self, Bucket_name):
        if Bucket_name in self.listBuckets():
            return True
        else:
            return False


    #Object management
    def listObjects(self, Bucket_name):
        return self.connection_with_minio.list_objects(Bucket_name)

    def deleteObjects(self, Bucket_name, Object_name):
        self.connection_with_minio.remove_object(Bucket_name, Object_name )