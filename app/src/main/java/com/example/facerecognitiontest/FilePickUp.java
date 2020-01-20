package com.example.facerecognitiontest;

import androidx.appcompat.app.AppCompatActivity;

import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.android.volley.Request;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.example.facerecognitiontest.facenet.FaceNet;
import com.example.facerecognitiontest.facenet.FaceNetMobile;
import com.example.facerecognitiontest.facenet.MobileFaceNet;
import com.example.facerecognitiontest.util.MyUtil;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

import static com.example.facerecognitiontest.App.list_label_mobile;
import static com.example.facerecognitiontest.App.list_label_pb;
import static com.example.facerecognitiontest.App.list_label_tf;
import static com.example.facerecognitiontest.App.list_vector_mobile;
import static com.example.facerecognitiontest.App.list_vector_pb;
import static com.example.facerecognitiontest.App.list_vector_tf;
import static com.example.facerecognitiontest.facenet.FaceNet.EMBEDDING_SIZE;

public class FilePickUp extends AppCompatActivity {

    JSONObject json = new JSONObject();
    private static final String TAG = "testing";
    private MobileFaceNet mfn;
    ImageView image;
    private static final int FACE_SIZE = 160;
    FaceNetMobile face = new FaceNetMobile();
    private FaceNet faceNet;
    TextView mobileView, pbView, tfView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_file_pick_up);

        image = findViewById(R.id.image);
        mobileView = findViewById(R.id.mobile);
        pbView = findViewById(R.id.pb);
        tfView = findViewById(R.id.tf);

        try {
            mfn = new MobileFaceNet(getAssets());
            copyBigDataToSD("det1.bin");
            copyBigDataToSD("det2.bin");
            copyBigDataToSD("det3.bin");
            copyBigDataToSD("det1.param");
            copyBigDataToSD("det2.param");
            copyBigDataToSD("det3.param");
            copyBigDataToSD("recognition.bin");
            copyBigDataToSD("recognition.param");
        } catch (IOException e) {
            e.printStackTrace();
        }
        //model init
        File sdDir = Environment.getExternalStorageDirectory();//get directory
        String sdPath = sdDir.toString() + "/test/";
        face.FaceModelInit(sdPath);
        faceNet = FaceNet.create(getAssets(), FACE_SIZE, FACE_SIZE);


        String path = Environment.getExternalStorageDirectory().toString()+"/Test_tflite/infer";
        Log.d("Files", "Path: " + path);
        File directory = new File(path);
        File[] files = directory.listFiles();
        Log.d("Files", "Size: "+ files.length);

        for (int i=0;i<files.length;i++) {
            Log.i("testing_path",files[i].getName());
            try {
                json.put("Name",files[i].getName());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            Bitmap b = BitmapFactory.decodeFile(files[i].getAbsolutePath());
            ImageView image = findViewById(R.id.image);
            find_distance(b);
            image.setImageBitmap(b);
        }
    }

    private void find_distance(Bitmap bitmap) {
        image.setImageBitmap(bitmap);
        int faceInfo[] = null;
        try {
            byte[] imageDate = getPixelsRGBA(bitmap);
            faceInfo = face.FaceDetect(imageDate, bitmap.getWidth(), bitmap.getHeight(), 4);
            Log.i("test", faceInfo[0] + "");
            Rect rect = new Rect(faceInfo[1 + 14 * 0], faceInfo[2 + 14 * 0], faceInfo[3 + 14 * 0], faceInfo[4 + 14 * 0]);
            mobileTest(rect, bitmap);
            tfTest(rect, bitmap);
            pbTest(rect, bitmap);
            Log.i("test_json",json.toString());
            sendApi();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private void sendApi(){

        JsonObjectRequest myReq = new JsonObjectRequest(Request.Method.POST,
                "http://192.168.3.227:8080/json-example",json,
                response -> {
            Log.i("testing_respose",response.toString());
                }, error -> {
            Log.i("testing_respose",error.toString());
        });
        Volley.newRequestQueue(this).add(myReq);
    }

    private void pbTest(Rect rect, Bitmap bitmap) throws JSONException {
        JSONObject pbjson = new JSONObject();

        float[] emb_array = new float[EMBEDDING_SIZE];
        faceNet.getEmbeddings(bitmap, rect).get(emb_array);

        float[] inf = emb_array;

        String text = "\n";
        float sum = 0;
        double dist = 0;
        int k = 0;

        for (int i = 0; i < list_vector_pb.size(); i++) {
            if (i % 15 == 0) {
                if (dist != 0) {
                    pbjson.put(list_label_mobile.get(k-1),(dist / 15));
                    text += (dist / 15) + "\n";
                    dist = 0;
                }
                text += list_label_pb.get(k) + ":-   ";
                k++;
            }
            float[] reg = list_vector_pb.get(i);
            sum = 0;
            for (int j = 0; j < reg.length; j++) {
                sum += Math.pow((reg[j] - inf[j]), 2);
            }
            dist += Math.sqrt(sum);
        }
        pbjson.put(list_label_mobile.get(k-1),(dist / 15));
        text += (dist / 15) + "\n\n";
        pbView.append(text);
        json.put("pb",pbjson);
    }

    private void tfTest(Rect rect, Bitmap bitmap) throws JSONException {
        JSONObject tfjson = new JSONObject();
        Bitmap b = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
        byte[] faceDate1 = getPixelsRGBA(b);
        float[] embFeature1 = face.FaceEmbFeature(faceDate1, b.getWidth(), b.getHeight());
        float[] inf = embFeature1;

        String text = "\n";
        float sum = 0;
        double dist = 0;
        int k = 0;
        for (int i = 0; i < list_vector_tf.size(); i++) {
            if (i % 15 == 0) {
                if (dist != 0) {
                    tfjson.put(list_label_mobile.get(k-1),(dist / 15));
                    text += (dist / 15) + "\n";
                    dist = 0;
                }
                text += list_label_tf.get(k) + ":-   ";
                k++;
            }
            float[] reg = list_vector_tf.get(i);
            sum = 0;
            for (int j = 0; j < reg.length; j++) {
                sum += Math.pow((reg[j] - inf[j]), 2);
            }
            dist += Math.sqrt(sum);
        }

        tfjson.put(list_label_mobile.get(k-1),(dist / 15));

        text += (dist / 15) + "\n\n";
        tfView.append(text);
        json.put("tf",tfjson);
    }

    private void mobileTest(Rect rect, Bitmap bitmap) throws JSONException {
        JSONObject mobilejson = new JSONObject();
        Bitmap bitmapCrop1 = MyUtil.crop(bitmap, rect);
        float[][] embeddings = mfn.compare(bitmapCrop1, bitmapCrop1);
        float[] inf = embeddings[0];

        String text = "\n";
        float sum = 0;
        double dist = 0;
        int k = 0;

        for (int i = 0; i < list_vector_mobile.size(); i++) {
            if (i % 15 == 0) {
                if (dist != 0) {
                    mobilejson.put(list_label_mobile.get(k-1),(dist / 15));
                    text += (dist / 15) + "\n";
                    dist = 0;
                }

                text += list_label_mobile.get(k) + ":-   ";
                k++;
            }
            float[] reg = list_vector_mobile.get(i);
            sum = 0;
            for (int j = 0; j < reg.length; j++) {
                sum += Math.pow((reg[j] - inf[j]), 2);
            }
            dist += Math.sqrt(sum);
        }
        mobilejson.put(list_label_mobile.get(k-1),(dist / 15));
        text += (dist / 15) + "\n\n";
        mobileView.append(text);
        json.put("mobile",mobilejson);
    }

    private byte[] getPixelsRGBA(Bitmap image) {
        // calculate how many bytes our image consists of
        int bytes = image.getByteCount();
        ByteBuffer buffer = ByteBuffer.allocate(bytes); // Create a new buffer
        image.copyPixelsToBuffer(buffer); // Move the byte data to the buffer
        byte[] temp = buffer.array(); // Get the underlying array containing the

        return temp;
    }

    private void copyBigDataToSD(String strOutFileName) throws IOException {
        Log.i(TAG, "start copy file " + strOutFileName);
        File sdDir = Environment.getExternalStorageDirectory();//get directory
        File file = new File(sdDir.toString() + "/test/");
//        LOGGER.i("wjy debug " + sdDir.toString() + "/facem/" );    //wjy
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = sdDir.toString() + "/test/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.i(TAG, "file exists " + strOutFileName);
            return;
        }
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString() + "/test/" + strOutFileName);
        myInput = this.getAssets().open(strOutFileName);
        byte[] buffer = new byte[1024];
        int length = myInput.read(buffer);
        while (length > 0) {
            myOutput.write(buffer, 0, length);
            length = myInput.read(buffer);
        }
        myOutput.flush();
        myInput.close();
        myOutput.close();
        Log.i(TAG, "end copy file " + strOutFileName);
    }
}