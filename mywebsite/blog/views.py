from django.shortcuts import render
from blog.models import Category, Post
from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import CreateView
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

    
# Create your views here.
def home(request):
    return render(request,'home.html')

def bitcoin(request):
    ed = request.POST['ed']
    ed = float(ed)
    bcc = request.POST['bcc']
    bcc = float(bcc)
    eos = request.POST['eos']
    eos = float(eos)
    tron = request.POST['tron']
    tron = float(tron)
    lc = request.POST['lc']
    lc = float(lc)
    tf.compat.v1.disable_eager_execution()
    X = tf.placeholder(tf.float32, shape=[None, 5])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.random_normal([5, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")
    hypothesis = tf.matmul(X, W) + b
    saver = tf.train.Saver()
    model = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(model)
        save_path = "./saved.cpkt"
        saver.restore(sess, save_path)
        data = ((ed, bcc, eos, tron, lc),)
        arr = np.array(data, dtype=np.float32)
        x_data = arr[0:5]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
    bitcoin = dict[0]
    return render(request,'bitcoin.html',{'ed':ed,'bcc':bcc,'eos':eos,'tron':tron,'lc':lc,'bitcoin':bitcoin })


# Create your views here.
def index(req):
    post_latest = Post.objects.order_by("-createDate")[:6]
    context = {
        "post_latest": post_latest
    }

    return render(req, "index.html", context=context)

class PostDetailView(generic.DetailView):
    model = Post

class PostCreate(LoginRequiredMixin, CreateView):
    model = Post
    fields = ["title", "title_image", "content", "category"]