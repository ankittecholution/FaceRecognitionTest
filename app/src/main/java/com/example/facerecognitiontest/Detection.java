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
import android.util.JsonReader;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.example.facerecognitiontest.facenet.FaceNet;
import com.example.facerecognitiontest.facenet.FaceNetMobile;
import com.example.facerecognitiontest.facenet.MobileFaceNet;
import com.example.facerecognitiontest.util.FileUtils;
import com.example.facerecognitiontest.util.MyUtil;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;

import static com.example.facerecognitiontest.App.list_label_mobile;
import static com.example.facerecognitiontest.App.list_label_pb;
import static com.example.facerecognitiontest.App.list_label_tf;
import static com.example.facerecognitiontest.App.list_vector_mobile;
import static com.example.facerecognitiontest.App.list_vector_pb;
import static com.example.facerecognitiontest.App.list_vector_tf;
import static com.example.facerecognitiontest.facenet.FaceNet.EMBEDDING_SIZE;

public class Detection extends AppCompatActivity {

    JSONObject json = new JSONObject();

    private static final String TAG = "testing";

    private MobileFaceNet mfn;
    ImageView image;

    private static final int FACE_SIZE = 160;

    FaceNetMobile face = new FaceNetMobile();

    private FaceNet faceNet;

    TextView mobileView,pbView,tfView;

    long sum1= 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);
        image = findViewById(R.id.image);
        mobileView= findViewById(R.id.mobile);
        pbView= findViewById(R.id.pb);
        tfView= findViewById(R.id.tf);
        image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                performFileSearch(10);
            }
        });
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

    private void find_distance(Bitmap bitmap) throws JSONException {
        image.setImageBitmap(bitmap);
        int faceInfo[] = null;
        byte[] imageDate = getPixelsRGBA(bitmap);
        faceInfo = face.FaceDetect(imageDate, bitmap.getWidth(), bitmap.getHeight(), 4);
        Log.i("test",faceInfo[0]+"");
        if (faceInfo[0]>0) {
            Rect rect = new Rect(faceInfo[1 + 14 * 0], faceInfo[2 + 14 * 0], faceInfo[3 + 14 * 0], faceInfo[4 + 14 * 0]);
            sum1 = 0;
            mobileTest(rect, bitmap);
//            Log.i("testing_time_mobile",(sum1/15)+"");
            sum1 = 0;
            tfTest(rect, bitmap);
//            Log.i("testing_time_mobile",(sum1/15)+"");
            sum1 = 0;
            pbTest(rect, bitmap);
//            Log.i("testing_time_mobile",(sum1/15)+"");
            sendApi();
        }
    }


    private void pbTest(Rect rect, Bitmap bitmap) throws JSONException {
        JSONObject pbjson = new JSONObject();

        long time = System.currentTimeMillis();
        float[] emb_array = new float[EMBEDDING_SIZE];
        faceNet.getEmbeddings(bitmap, rect).get(emb_array);
        long totaltime = System.currentTimeMillis()-time;
        Log.i("testing_time"," pb "+(totaltime)+"");

        sum1 += totaltime;

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
                sum += Math.pow((reg[j] - emb_array[j]), 2);
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

        long time = System.currentTimeMillis();
        Bitmap b = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
        byte[] faceDate1 = getPixelsRGBA(b);
        float[] embFeature1 = face.FaceEmbFeature(faceDate1, b.getWidth(), b.getHeight());
        long totaltime = System.currentTimeMillis()-time;
        Log.i("testing_time"," TF "+(totaltime)+"");

        sum1+=totaltime;
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

        long time = System.currentTimeMillis();
        Bitmap bitmapCrop1 = MyUtil.crop(bitmap, rect);
        float[][] embeddings = mfn.compare(bitmapCrop1, bitmapCrop1);
        float[] inf = embeddings[0];

        long totaltime = System.currentTimeMillis()-time;
        Log.i("testing_time"," mobile "+(totaltime)+"");

        sum1+=totaltime;

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

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            ClipData clipData = data.getClipData();
            if (clipData == null) {
                try {
                    find_distance(getBitmapFromUri(getContentResolver(),data.getData()));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }


    private Bitmap getBitmapFromUri(ContentResolver contentResolver, Uri uri) throws Exception {
        ParcelFileDescriptor parcelFileDescriptor =
                contentResolver.openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();

        return bitmap;
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
        File file = new File(sdDir.toString()+"/test/");
//        LOGGER.i("wjy debug " + sdDir.toString() + "/facem/" );    //wjy
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = sdDir.toString()+"/test/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.i(TAG, "file exists " + strOutFileName);
            return;
        }
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString()+"/test/"+ strOutFileName);
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

    public void performFileSearch(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setType("image/*");

        startActivityForResult(intent, requestCode);
    }
}
