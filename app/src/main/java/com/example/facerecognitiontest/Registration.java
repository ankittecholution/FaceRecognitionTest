package com.example.facerecognitiontest;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.app.Activity;
import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.example.facerecognitiontest.facenet.FaceNet;
import com.example.facerecognitiontest.facenet.FaceNetMobile;
import com.example.facerecognitiontest.facenet.MobileFaceNet;
import com.example.facerecognitiontest.util.FileUtils;
import com.example.facerecognitiontest.util.MyUtil;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;

import static com.example.facerecognitiontest.facenet.FaceNet.EMBEDDING_SIZE;

public class Registration extends AppCompatActivity {

    private static final String TAG = "testing";
    private MobileFaceNet mfn;
    private static final int FACE_SIZE = 160;

    TextView text;

    Button click;

    FaceNetMobile face = new FaceNetMobile();

    private FaceNet faceNet;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;

    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.CAMERA",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        verifyStoragePermissions(this);

        setTitle("Registration");

        text = findViewById(R.id.name);
        click = findViewById(R.id.click);


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

        init();
        click.setOnClickListener(v -> performFileSearch(10));

    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            ClipData clipData = data.getClipData();
            ArrayList<Uri> uris = new ArrayList<>();
            int faceInfo[] = null;

            if (clipData == null) {
                uris.add(data.getData());
            } else {
                ArrayList<float[]> pb_list = new ArrayList<>();
                ArrayList<float[]> tf_list = new ArrayList<>();
                ArrayList<float[]> mobile_list = new ArrayList<>();
                for (int i = 0; i < clipData.getItemCount(); i++) {
                    try {
                        Bitmap b = getBitmapFromUri(getContentResolver(),clipData.getItemAt(i).getUri());
                        byte[] imageDate = getPixelsRGBA(b);
                        faceInfo = face.FaceDetect(imageDate, b.getWidth(), b.getHeight(), 4);
                        Log.i(TAG, "onActivityResult: "+faceInfo[0]);
                        if (faceInfo[0] == 1) {
                            Rect rect = new Rect(faceInfo[1 + 14 * 0], faceInfo[2 + 14 * 0], faceInfo[3 + 14 * 0], faceInfo[4 + 14 * 0]);
                            float[] emb_array = new float[EMBEDDING_SIZE];
                            faceNet.getEmbeddings(b, rect).get(emb_array);
                            pb_list.add(emb_array);
                            Bitmap bitmap = Bitmap.createBitmap(b, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
                            byte[] faceDate1 = getPixelsRGBA(bitmap);
                            float[] embFeature1 = face.FaceEmbFeature(faceDate1, bitmap.getWidth(), bitmap.getHeight());
                            tf_list.add(embFeature1);
                            Bitmap bitmapCrop1 = MyUtil.crop(b, rect);
                            float[][] embeddings = mfn.compare(bitmapCrop1, bitmapCrop1);
                            mobile_list.add(embeddings[0]);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    uris.add(clipData.getItemAt(i).getUri());
                }
                if (mobile_list.size()==15) {
                    storeVector(pb_list, FileUtils.DATA_FILE_PB);
                    storeVector(tf_list, FileUtils.DATA_FILE_TF);
                    storeVector(mobile_list, FileUtils.DATA_FILE_MOBILE);
                }else {
                    Toast.makeText(this,"Please choose only 15 images  "+mobile_list.size(),Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    private void storeVector(ArrayList<float[]> list,String s){
        StringBuilder builder = new StringBuilder();
        for (int k = 0; k < list.size(); k++) {
            float[] array = list.get(k);
            builder.append(text.getText().toString());
            for (int j = 0; j < array.length; j++) {
                builder.append(" ").append(j).append(":").append(array[j]);
            }
            if (k < list.size() - 1) builder.append(System.lineSeparator());
        }
        FileUtils.appendText(builder.toString(), s);
    }


    void init() {
        File dir = new File(FileUtils.ROOT);

        if (!dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();

            AssetManager mgr = getAssets();
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE_PB);
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE_MOBILE);
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE_TF);
            FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
            FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
        }

    }

    public void performFileSearch(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setType("image/*");

        startActivityForResult(intent, requestCode);
    }

    public static void verifyStoragePermissions(Activity activity) {

        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
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



}
